import time
import random
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from .model import RNN, device

class NameGenerator():
    def __init__(self, dataset: IterableDataset, epochs=10000, hidden_size=64, lr=0.03) -> None:
        self.dataset = dataset
        self.chunk_len = 12 # how many characters at a time
        # self.num_epochs = 6768 # number of epochs to train for --> chosen because it's the number of sequences
        self.num_epochs = epochs
        self.batch_size = 1
        self.print_every = 1000
        self.plot_every = 100
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.lr = lr
    
    def get_random_batch(self) -> torch.Tensor:
        """
        Get a random batch from the training data

        Returns:
            tuple: (input, target)
        """
        return self.dataset[random.randint(0, self.dataset.num_features - self.chunk_len)]
    
    def generate(self, initial_str='a', predict_len=100, temperature=0.85):
        """
        Generate some names using the model
        """
        # make initial_str a random lowercase char a to z
        # initial_str = random.choice(string.ascii_lowercase)
        
        # make prediction length a random number between 6 and 12 and ensure it is less than the chunk length
        # predict_len = random.randint(6, 12)
        # assert predict_len <= self.chunk_len, "Prediction length must be less than chunk length" # Does it?
        
        hidden, cell = self.rnn.init_hidden(self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        # Use priming string to "build up" hidden state if initial_str is longer than 1
        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)
        
        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            # predicted_char = idx2char[top_char.item()]
            predicted_char = self.dataset.detensorize(top_char.item())
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        # TODO: clean this up to remove partial names
        return predicted

    def train(self):
        """
        Train the model
        """
        losses = []

        # input_size, hidden_size, num_layers, output_size
        self.rnn = RNN(self.dataset.num_chars, self.hidden_size, self.num_layers, self.dataset.num_chars).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print(" -> Starting training...")
        start = time.time()
        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:,c], hidden, cell)
                loss += criterion(output, target[:,c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len
            
            # loss tracking
            losses.append(loss)

            if epoch % self.print_every == 0:
                print(f"Epoch: {epoch}/{self.num_epochs} Loss: {loss}")
        print(" -> Training complete!")
        end = time.time()
        print(f" -> Total training time: {end - start}s")
        print(f"Max loss: {max(losses)}")
        print(f"Min loss: {min(losses)}")

        return losses
    
    def load_model(self, model_path):
        self.rnn = torch.load(model_path)

    def save_model(self, model_path):
        torch.save(self.rnn, model_path)