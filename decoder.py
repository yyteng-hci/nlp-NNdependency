from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import numpy as np
import tensorflow as tf
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            #pass
            # TODO: Write the body of this loop for part 4
            # use the feature extractor to obtain a representation of the current state
            features = self.extractor.get_input_representation(words, pos, state).reshape((1, 6))
            # call model.predict(features) and retrieve a softmax actived vector of possible actions.
            predict = self.model.predict(features)[0]
            #print(predict)
            # create a list of possible actions
            # sort it according to output probability
            actions = list(np.argsort(predict)[::-1])
            #print(actions)
            # take the largest probability legal transition
            for i in actions:
                arc,dep = self.output_labels[i]
                if arc == 'shift':
                    if not state.stack:
                        state.shift()
                        break
                    elif len(state.buffer) > 1:
                        state.shift()
                        break
                elif arc == 'left_arc':
                    if state.stack and state.stack[-1] != 0:
                        state.left_arc(dep)
                        break
                elif arc == 'right_arc':
                    if state.stack:
                        state.right_arc(dep)
                        break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'
    tf.compat.v1.disable_eager_execution()

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
