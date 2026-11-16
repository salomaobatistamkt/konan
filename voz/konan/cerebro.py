"""Conversa com o llama.cpp (API OpenAI) em streaming, frase por frase, com a ferramenta de matemática."""
import json
import re
import time
from datetime import datetime
from typing import Callable, Iterator

import httpx
from urllib.parse import urlparse

from . import config, matematica

SISTEMA = """Você é {nome}, uma assistente de voz. Você conversa com uma pessoa de verdade, em português do Brasil.
Fale como uma brasileira educada falaria numa conversa comum: natural, clara, calorosa e direta. Nem formal demais, nem gíria.
Use palavras normais do dia a dia: "estou", "está", "tudo bem", "certo", "claro", "combinado". Não use "tô", "tá", "suave", "beleza", "mano", "cara", "valeu" nem abreviações de fala.
Suas respostas serão FALADAS em voz alta, então:
- Responda curto: 1 a 3 frases, a não ser que peçam explicação. Comece pela resposta, sem rodeios.
- Nada de markdown, listas, emojis, tabelas ou código. Escreva números como se fala.
- Entenda o que a pessoa quis dizer mesmo que a fala venha com erros da transcrição. Se realmente não entender, peça para repetir de forma simples.
- Para QUALQUER conta ou problema de matemática use a ferramenta calcular e confie no resultado dela; nunca invente números.
- Se não souber algo, diga com sinceridade que não sabe.
Exemplos do seu jeito de falar:
Usuário: que horas são? / Você: Agora são três e meia da tarde.
Usuário: como você está? / Você: Estou bem, obrigada por perguntar. E você, como está o seu dia?
Usuário: preciso de uma ideia de jantar / Você: Que tal um frango grelhado com arroz e salada? É rápido e leve.
Hoje é {hoje}.
{fatos}{memorias}"""

_DIAS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
_FIM_FRASE = re.compile(r"(.+?[.!?…:;])(\s+|$)", re.S)


def montar_sistema(memorias: list[tuple[float, str, str]], fatos: list[str] | None = None) -> str:
    agora = datetime.now()
    hoje = f"{_DIAS[agora.weekday()]}, {agora:%d/%m/%Y}, {agora:%H:%M}"
    bloco = ""
    if memorias:
        linhas = [f'- em {datetime.fromtimestamp(q):%d/%m}: usuário disse "{u}" e você respondeu "{k}"' for q, u, k in memorias]
        bloco = "Lembranças de conversas antigas que podem ser úteis:\n" + "\n".join(linhas)
    bloco_fatos = ("O que você sabe sobre o usuário:\n" + "\n".join(f"- {f}" for f in fatos) + "\n") if fatos else ""
    return SISTEMA.format(nome=config.NOME, hoje=hoje, memorias=bloco, fatos=bloco_fatos)


_SIMBOLOS = [
    (r"\*\*|__|`+|#+\s*", ""), (r"\$+", ""), (r"\\\(|\\\)|\\\[|\\\]", ""),
    (r"√", " raiz de "), (r"π", " pi "), (r"≈", " aproximadamente "), (r"≠", " diferente de "),
    (r"≤", " menor ou igual a "), (r"≥", " maior ou igual a "), (r"×", " vezes "), (r"÷", " dividido por "),
    (r"²", " ao quadrado"), (r"³", " ao cubo"), (r"°", " graus"), (r"(?<=\d)\s*%", " por cento"),
    (r"(?<=\w)\^(\d+)", r" elevado a \1"), (r"^\s*[-*•]\s+", ""),
]


def para_fala(texto: str) -> str:
    """Limpa o texto para soar bem na voz."""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    for padrao, troca in _SIMBOLOS:
        texto = re.sub(padrao, troca, texto, flags=re.M)
    return re.sub(r"\s{2,}", " ", texto).strip()



def url_local(url: str) -> bool:
    """True se o cérebro roda na própria máquina/rede (nome de serviço do Docker, localhost ou IP privado)."""
    host = (urlparse(url).hostname or "").lower()
    return (
        "." not in host or host == "localhost" or host.startswith(("127.", "10.", "192.168."))
        or bool(re.match(r"172\.(1[6-9]|2\d|3[01])\.", host))
    )


class Cerebro:
    def __init__(self, url: str | None = None, cliente: httpx.Client | None = None,
                 chave: str | None = None, modelo: str | None = None):
        self.base = (url or config.CEREBRO_URL).rstrip("/")
        self.url = self.base + "/chat/completions"
        self.chave = config.CEREBRO_CHAVE if chave is None else chave
        self.modelo = modelo or config.CEREBRO_MODELO
        cabecalhos = {"Authorization": f"Bearer {self.chave}"} if self.chave else {}
        self.cliente = cliente or httpx.Client(timeout=httpx.Timeout(120, connect=5))
        self.cliente.headers.update(cabecalhos)
        # Campos extras só existem no llama.cpp; APIs externas (Groq/Gemini) podem recusá-los.
        self.local = not self.chave or url_local(self.base)
        self.ativos = 0  # conversas em andamento (o aprendiz espera ficar em 0)

    def completar(self, sistema: str, texto: str, max_tokens: int = 160) -> str:
        """Pergunta curta, sem streaming (usada para extrair fatos)."""
        corpo = {"model": self.modelo, "temperature": 0, "max_tokens": max_tokens, "stream": False,
                 "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": texto}]}
        if self.local:
            corpo["chat_template_kwargs"] = {"enable_thinking": False}
        self.ativos += 1
        try:
            r = self.cliente.post(self.url, json=corpo)
            r.raise_for_status()
            return re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"].get("content") or "", flags=re.S)
        finally:
            self.ativos -= 1

    def pronto(self) -> bool:
        try:
            if self.local:
                raiz = self.base[:-3] if self.base.endswith("/v1") else self.base
                return self.cliente.get(raiz + "/health", timeout=3).status_code == 200
            return self.cliente.get(self.base + "/models", timeout=10).status_code == 200
        except httpx.HTTPError:
            return False

    def _stream(self, mensagens: list[dict], com_ferramentas: bool) -> Iterator[dict]:
        corpo = {
            "model": self.modelo,
            "messages": mensagens,
            "stream": True,
            "temperature": config.CEREBRO_TEMPERATURA,
            "max_tokens": config.CEREBRO_MAX_TOKENS,
        }
        if self.local:
            corpo["cache_prompt"] = True
            corpo["chat_template_kwargs"] = {"enable_thinking": False}
        if com_ferramentas:
            corpo["tools"] = [matematica.FERRAMENTA]
        with self.cliente.stream("POST", self.url, json=corpo) as resp:
            resp.raise_for_status()
            for linha in resp.iter_lines():
                if not linha.startswith("data:"):
                    continue
                dado = linha[5:].strip()
                if dado == "[DONE]":
                    break
                try:
                    escolha = json.loads(dado)["choices"][0]
                except (ValueError, KeyError, IndexError):
                    continue
                yield escolha.get("delta") or {}

    def responder(self, mensagens: list[dict], ao_calcular: Callable[[str, str], None] | None = None) -> Iterator[str]:
        """Gera a resposta frase por frase (para a voz começar a falar logo)."""
        self.ativos += 1
        try:
            yield from self._responder(mensagens, ao_calcular)
        finally:
            self.ativos -= 1

    def _responder(self, mensagens, ao_calcular):
        mensagens = list(mensagens)
        for rodada in range(4):
            chamadas: dict[int, dict] = {}
            buffer, dentro_think = "", False
            for delta in self._stream(mensagens, com_ferramentas=rodada < 3):
                for tc in delta.get("tool_calls") or []:
                    item = chamadas.setdefault(tc.get("index", 0), {"id": "", "nome": "", "args": ""})
                    item["id"] = tc.get("id") or item["id"]
                    fn = tc.get("function") or {}
                    item["nome"] += fn.get("name") or ""
                    item["args"] += fn.get("arguments") or ""
                pedaco = delta.get("content") or ""
                if not pedaco:
                    continue
                buffer += pedaco
                if "<think>" in buffer:
                    dentro_think = True
                if dentro_think:
                    if "</think>" not in buffer:
                        continue
                    buffer = buffer.split("</think>", 1)[1]
                    dentro_think = False
                if chamadas:
                    continue
                while (m := _FIM_FRASE.match(buffer)) and len(m.group(1).strip()) > 1:
                    frase = para_fala(m.group(1))
                    buffer = buffer[m.end():]
                    if frase:
                        yield frase
            if not chamadas:
                resto = para_fala(buffer)
                if resto:
                    yield resto
                return
            # Executa as contas pedidas e devolve ao modelo
            lista = [chamadas[i] for i in sorted(chamadas)]
            mensagens.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": c["id"] or f"c{n}", "type": "function", "function": {"name": c["nome"], "arguments": c["args"]}}
                               for n, c in enumerate(lista)],
            })
            for n, c in enumerate(lista):
                try:
                    expressao = json.loads(c["args"] or "{}").get("expressao", "")
                except ValueError:
                    expressao = c["args"]
                resultado = matematica.calcular(expressao) if c["nome"] == "calcular" else "erro: ferramenta desconhecida"
                if ao_calcular:
                    ao_calcular(expressao, resultado)
                mensagens.append({"role": "tool", "tool_call_id": c["id"] or f"c{n}", "content": resultado})
        yield "Não consegui fechar essa conta. Pode repetir de outro jeito?"


def esperar_cerebro(cerebro: Cerebro, avisar: Callable[[str], None]) -> None:
    inicio = time.time()
    while not cerebro.pronto():
        if time.time() - inicio > 5:
            avisar("aguardando o cérebro responder em %s (na primeira vez o modelo local é baixado; com API, confira a chave)…" % cerebro.base)
            inicio = time.time() + 55
        time.sleep(2)
