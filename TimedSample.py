import time
from typing import TypeVar

T = TypeVar('T')


class TimedSampleStore:

    def __init__(self, millis: int):
        self.__samples = []
        self.__millisToStore = millis

    def __removeOld(self):
        minTime = round(time.time() * 1000) - self.__millisToStore
        self.__samples = self.get_samples(minTime)

    @property
    def storedMillis(self) -> int:
        return self.__millisToStore

    def add(self, item: T):
        t = round(time.time() * 1000)
        self.__samples.append((t, item))
        self.__removeOld()

    def get_samples(self, sinceMillis: int = 0) -> list[(int, T)]:
        if sinceMillis == 0:
            # default value uses all stored samples
            self.__removeOld()
            return self.__samples
        else:
            return [x for x in self.__samples if x[0] >= sinceMillis]

    def count(self, sinceMillis: int = 0) -> int:
        self.__removeOld()
        if sinceMillis == 0:
            # default value uses all stored samples
            return len(self.__samples)
        else:
            return len(self.get_samples(sinceMillis))