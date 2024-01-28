import torch
import pandas as pd
from torch.utils.data import IterableDataset # Dataset
import os

# the dataset is handling the encoding, so the model only needs to handle the decoding
# this also means that all inputs passed into the model will need to be decoded
# TODO: can the encoding before moved into the model to simplify the dataset?
class PokemonNameDataset(IterableDataset):
    def __init__(self):
        super(PokemonNameDataset, self).__init__()
        # load the data
        self.pokemon_names = self.load_data()

        # all pokemon concatenated together
        # used to predict the next character
        self.corpus = '\n'.join(self.pokemon_names).lower()

        # find all unique characters in the names using set
        self.chars = sorted(list(set(self.corpus)))
        self.num_chars = len(self.chars)

        # Build translation dictionaries, 'a' -> 0, 0 -> 'a'
        self.encoding = dict((c, i) for i, c in enumerate(self.chars))
        self.decoding = dict((i, c) for i, c in enumerate(self.chars))

        # Use longest name length as our sequence window
        self.sequence_length = max((len(name) for name in self.pokemon_names)) # max sequence length
        self.corpus_length = len(self.corpus)

        # self.X, self.Y = self.convert()
        self.feature_label_pairs = self.convert()
        self.num_features = len(self.feature_label_pairs)

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.feature_label_pairs[idx]

    def __iter__(self):
        return ([idx[0], idx[1]] for idx in self.feature_label_pairs)
    
    def encode(self, key):
        return self.encoding[key]
    
    def decode(self, key):
        return self.decoding[key]
    
    def tensorize(self, seq):
        return torch.tensor([self.encode(char) for char in seq])
    
    def detensorize(self, tensor):
        return ''.join([self.decode(idx) for idx in tensor])
    
    def load_data(self):
        pokemon_names = []
        # check to see if pokemon_names.csv exists
        if os.path.exists('/Users/waslow/Data/pokemon_names.csv'):
            # read the names into a list
            with open('/Users/waslow/Data/pokemon_names.csv', 'r') as f:
                pokemon_names = f.read().split('\n')
        else:
            # read the pokemon names csv into a df
            df = pd.read_csv('/Users/waslow/Data/pokemon.csv')
            pokemon_names = df['name'].tolist()

            # save the names to a file
            with open('/Users/waslow/Data/pokemon_names.csv', 'w') as f:
                f.write('\n'.join(pokemon_names))

        return pokemon_names

    def convert(self):
        raw_features = [] # sequences
        raw_labels = [] # next_chars
        step_length = 1 

        # slide a window of size sequence_length over the data
        # to create a training set
        for i in range(0, self.corpus_length - self.sequence_length, step_length):
            raw_features.append(self.corpus[i: i + self.sequence_length])
            raw_labels.append(self.corpus[i + self.sequence_length])

        # num_features = len(raw_features)

        # features = [tensorize(seq) for seq in raw_features]
        # labels = [tensorize(seq) for seq in raw_labels]
        return list(zip(
            [self.tensorize(seq) for seq in raw_features],
            [self.tensorize(seq) for seq in raw_labels]
        ))