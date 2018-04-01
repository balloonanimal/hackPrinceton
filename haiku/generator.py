from nltk.corpus import words
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from random import *
"""
Notes:
    General flow of program will function as follows:
     - Call the functions gen(n)
     - gen(n) will create a sentence of n syllables
       - gen(n) will first create a sentence structure to its furthest non-terminal point.
       - It will then assign words to the variables of this structured sentence, starting by
         randomly choosing one element of the sentence (eg verb) and choosing any random word
         from the nltk.corpus cmudict. The templatic element will then be replaced by this specific
         instance, and n (the target number of syllables in the sentence) will be decremented by
         the number of syllables in this instance. Then, another templatic element will be randomly
         chosen and the same process will take place, within a while loop to check that no word
         in the senetence is longer than the remaining syllable allotment for the sentence. If the
         last templatic element of the sentence is being chosen, only a word equal to the remaining
         syllables will be chosen (unless after some large num of choices none is the correct length,
         indicating that no such word exists). By randomly choosing the element of the sentence,
         the non-rigid structure of the haiku will (hopefully) be simulated.
"""
class Generator:
    def __init__(self):
        self.cmu_d = cmudict.dict()
        self.my_dict = {}

        def populatePOS(dict_dict):
            part_to_speech = ["nouns","verbs","adjectives","adverbs","articles"]
            for pos in part_to_speech:
                list_of_this_part = do_populating(pos)
                dict_dict[pos[:len(pos)-1]] = list_of_this_part

        def do_populating(pos):
            filename = "haiku/words/" + pos + ".txt"
            word_list = []
            with open(filename) as in_file:
                data = in_file.read()
            in_file.close()
            spaced_data = data.replace("\n", " ")
            word_list = [word.strip() for word in spaced_data.split(" ")]
            return word_list[:len(word_list)-1]

        populatePOS(self.my_dict)

    def write_haiku(self):
        line1 = self.write_line(5) + "\n"
        line2 = self.write_line(7) + "\n"
        line3 = self.write_line(5)
        haiku = line1 + line2 + line3

        no_dash = haiku.replace("-", "")

        return no_dash

    def write_line(self, target_syll):
        templatic_sentence = self.make_sentence()
        literal_sentence = ""
        for index, pos_elem in enumerate(templatic_sentence):
            if target_syll > 0:
                still_deciding = True
                while still_deciding:
                    word_to_add = self.my_dict[pos_elem][randrange(len(self.my_dict[pos_elem]))]
                    syllables = self.syllable_count(word_to_add)
                    if (syllables <= target_syll) and (index != len(templatic_sentence)-1):
                        still_deciding = False
                        target_syll -= syllables
                    elif index == len(templatic_sentence)-1:
                        while syllables != target_syll:
                            word_to_add = self.my_dict[pos_elem][randrange(len(self.my_dict[pos_elem]))]
                            syllables = self.syllable_count(word_to_add)
                        still_deciding = False

                literal_sentence += word_to_add + " "
        return literal_sentence

    def syllable_count(self, word):
        word_list = self.cmu_d[word]

        syl_count = 0

        for symbol in word_list[0]:
            if symbol[-1].isdigit():
                syl_count += 1

        return syl_count

    def make_sentence(self):

        def fill_in(grammar,  element, sentence):
            if element in grammar:
                pos_spec = grammar[element]
                spec = pos_spec[randrange(len(pos_spec))]
                if isinstance(spec, list):
                    for inner_spec in spec:
                        fill_in(grammar, inner_spec, sentence)
                else:
                    fill_in(grammar, spec, sentence)
            else:
                sentence.append(element)
        grammar = {}
        literals = {}
        S = "start"
        X = "noun phrase"
        Y = "verb phrase"
        R = "article"
        V = "verb"
        D = "adverb"
        N = "noun"
        A = "adjective"
        grammar[S] = [X,Y]
        grammar[X] = [[R,N],[N],[R,A,N]]
        grammar[Y] = [[V],[V,D]]

        sentence = []
        starter = grammar[S]
        for element in starter:
            fill_in(grammar, element, sentence)
        return sentence
