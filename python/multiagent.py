import numpy as np
from collections import Counter
import itertools
from random import randrange, sample


def random_insert_seq(lst, seq):
    insert_locations = sample(range(len(lst) + len(seq)), len(seq))
    inserts = dict(zip(insert_locations, seq))
    i = iter(lst)
    lst[:] = [inserts[pos] if pos in inserts else next(i)
              for pos in range(len(lst) + len(seq))]


class Agent:

    def __init__(self, lang_: str, lang2: str, p_lang2: float) -> None:
        self.lang1 = lang_
        self.movedF = False

        if np.random.random_sample() < p_lang2:
            self.lang2 = lang2
        else:
            self.lang2 = None

    def add_lang(self, nbhd, rho: float) -> None:
        if not self.bilingual():
            nbhd_langs = Counter(list(itertools.chain.from_iterable([n.get_langs() for n in nbhd]))).most_common()
            for (x, y) in nbhd_langs:
                y = y * rho
                if np.random.random_sample() < y:
                    if x != self.lang1:
                        self.lang2 = x

    def bilingual(self) -> bool:
        return self.lang2 is not None

    def get_langs(self):
        if self.bilingual():
            return [self.lang1, self.lang2]
        else:
            return [self.lang1]

    def moved_this_t(self):
        return self.movedF

    def move(self):
        self.movedF = True


class Country:

    def __init__(self, name: str, p_i: int, world_pop: int, cont: float, lang: str,
                 lang2: str, p_lang2: float, birth: float, death: float):
        self.lang = lang

        if lang2 == "None":
            self.lang2 = None
        else:
            self.lang2 = lang2

        self.p2 = p_lang2
        self.name = name
        self.cont = cont
        self.birth = birth
        self.death = death
        self.pop = [Agent(lang_=self.lang, lang2=self.lang2, p_lang2=self.p2) for _ in range(p_i)]
        self.util = (world_pop - self.get_pop())/world_pop

    def emmigrate(self, dest, rho):
        emis = []  # blessed be his name
        for agent in self.pop:
            if not agent.moved_this_t():
                if np.random.random_sample() < rho * self.cont:
                    agent.move()
                    emis.append(agent)
                    self.pop.remove(agent)

        random_insert_seq(dest.pop, emis)

    def educate(self, r, rho):
        for i, agent in enumerate(self.pop):
            if i <= r:
                agent.add_lang(self.pop[:i+r], rho)
            elif i >= len(self.pop)-1-r:
                agent.add_lang(self.pop[i-r:], rho)
            else:
                agent.add_lang(self.pop[i-r:i+r], rho)

    def do_birth(self):
        babies = [Agent(lang_=self.lang, lang2=self.lang2, p_lang2=self.p2) for _ in range(int(len(self.pop) * self.birth))]

        random_insert_seq(self.pop, babies)

    def do_death(self):
        n_dead = int(len(self.pop) * self.death)

        for i in range(n_dead):
            self.pop.pop(randrange(len(self.pop)))

    def unmove(self):
        for agent in self.pop:
            if agent.moved_this_t():
                agent.movedF = False

    def get_pop(self):
        return len(self.pop)

    def get_prilang(self):
        pri = []
        for agent in self.pop:
            pri.append(agent.get_langs()[0])
        return Counter(pri).most_common()

    def get_seclang(self):
        sec = []
        for agent in self.pop:
            if agent.bilingual():
                sec.append(agent.get_langs()[1])
        return Counter(sec).most_common()

    def get_spec_pri_lang(self, lang):
        pri = self.get_prilang()
        for (x, y) in pri:
            if x == lang:
                return y
        return 0

    def get_spec_sec_lang(self, lang):
        sec = self.get_seclang()
        for (x, y) in sec:
            if x == lang:
                return y
        return 0

    def update_util(self, world_pop: int):
        self.util = (world_pop - self.get_pop())/world_pop

    def census(self):
        pop = self.get_pop()
        prilangs = self.get_prilang()
        seclangs = self.get_seclang()

        print("{0}: Total population {1} million".format(self.name, pop))

        print("    Primary language data:")
        for (x, y) in prilangs:
            print("        {0}: {1} million speakers".format(x, y))

        print("    Secondary language data:")
        for (x, y) in seclangs:
            print("        {0}: {1} million speakers".format(x, y))

    def update_lang2(self):
        if self.get_seclang():
            (l2, n) = self.get_seclang()[0]
            self.lang2 = l2
            self.p2 = n/self.get_pop()
