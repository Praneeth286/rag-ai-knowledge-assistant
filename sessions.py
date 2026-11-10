"""Bounded, process-local session memory; no client-selected file paths."""
import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

@dataclass
class Session:
    history: list = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    touched: float = field(default_factory=time.monotonic)
    users: int = 0

class SessionBusy(Exception):
    pass

class Sessions:
    def __init__(self, capacity=256, ttl=3600, turns=10):
        self.items = {}
        self.capacity, self.ttl, self.turns = capacity, ttl, turns

    @asynccontextmanager
    async def use(self, key):
        now = time.monotonic()
        for old, entry in list(self.items.items()):
            if not entry.users and now - entry.touched > self.ttl:
                del self.items[old]
        if key not in self.items:
            if len(self.items) >= self.capacity:
                raise SessionBusy('Session capacity reached; retry later')
            self.items[key] = Session()
        entry = self.items[key]
        entry.users += 1
        try:
            async with entry.lock:
                yield entry.history
        finally:
            entry.touched = time.monotonic()
            entry.users -= 1

    def commit(self, history, question, answer):
        history.extend([('human', question), ('ai', answer)])
        del history[:-self.turns * 2]
