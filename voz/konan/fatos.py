"""Aprendizado em segundo plano: extrai fatos duráveis do que o usuário disse e guarda na memória.

Ideia adaptada do banco de memória do OpenJarvis (Apache-2.0): um extrator pede ao modelo uma lista curta
de fatos estáveis, a memória remove duplicados e mantém um teto. Aqui só roda quando a frase parece falar
do próprio usuário, e espera o cérebro ficar livre para não atrasar a conversa.
"""
import json
import logging
import queue
import re
import threading
import time

from . import config

log = logging.getLogger("konan.fatos")

_LEMBRAR = re.compile(r"\b(?:lembre|anote|guarde|memorize|n[ãa]o esque[cç]a)(?:-se)?(?:\s+(?:de|que|disso))?[:,]?\s+(.+)", re.I | re.S)
_GATILHO = re.compile(
    r"\b(meu|minha|meus|minhas|me chamo|eu (?:sou|moro|trabalho|gosto|tenho|odeio|adoro|prefiro|estudo|estou|vou|fa[cç]o)|"
    r"nasci|sou casad|sou solteir)\b", re.I)
_PEDIDO = (
    "Extraia da fala do usuário fatos duráveis sobre ele (nome, família, trabalho, cidade, gostos, rotina, projetos). "
    "Ignore perguntas, pedidos pontuais e conversa solta. Escreva cada fato em português, curto, na terceira pessoa "
    '(ex.: "Mora em Mossoró"). Responda SOMENTE com um array JSON de textos, ou [] se não houver nada.'
)


def extrair_explicito(texto: str) -> list[str]:
    """'Konan, lembre que minha filha se chama Ana' -> ['minha filha se chama Ana']."""
    m = _LEMBRAR.search(texto)
    return [m.group(1).strip().rstrip(".!")] if m and len(m.group(1).split()) >= 2 else []


def interpretar(saida: str) -> list[str]:
    m = re.search(r"\[.*\]", saida, re.S)
    if not m:
        return []
    try:
        itens = json.loads(m.group(0))
    except ValueError:
        return []
    return [i.strip() for i in itens if isinstance(i, str) and 4 <= len(i.strip()) <= 200][:3]


class Aprendiz:
    def __init__(self, memoria, cerebro):
        self.memoria, self.cerebro = memoria, cerebro
        self._fila: queue.Queue = queue.Queue(maxsize=8)
        threading.Thread(target=self._trabalhar, daemon=True).start()

    def submeter(self, pedido: str) -> None:
        if not config.FATOS_ATIVO:
            return
        try:
            self._fila.put_nowait(pedido)
        except queue.Full:
            pass

    def _trabalhar(self) -> None:
        while True:
            pedido = self._fila.get()
            try:
                fatos = extrair_explicito(pedido)
                if not fatos and _GATILHO.search(pedido):
                    while getattr(self.cerebro, "ativos", 0):  # deixa o cérebro livre para a conversa
                        time.sleep(0.5)
                    fatos = interpretar(self.cerebro.completar(_PEDIDO, pedido))
                for fato in fatos:
                    if self.memoria.guardar_fato(fato, config.FATOS_MAX):
                        log.info("aprendi: %s", fato)
            except Exception:  # noqa: BLE001
                log.debug("não consegui extrair fatos", exc_info=True)
