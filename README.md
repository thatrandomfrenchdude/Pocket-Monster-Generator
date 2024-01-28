## Pocket Monster Creator
Creates pokemon style pocket monsters with pokemon-style stats, moves, abilities, and descriptions based on a provided description.

### Inputs
Text Instruction
prompt: str as a Tensor

Text + Image Instruction
prompt: str and open-cv image: numpy.ndarray
combined as a Tensor

### Outputs
Pokemon: png
Moveset: str
Type(s): str[]
Pokedex: str
Stats: np.array

Flow:
Prompt -->
    Character Generator
    Movelist Generator

## Training Data
Kaggle Pokemon [Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon?resource=download)

## Components

### CharLSTM Pokemon Name Generator - In Development
Trained on existing Pokemon names

### Stable Diffusion Character generator - Not Yet Implemented
LoRA FineTuned on The Pokemon Dataset

### LLM Move Generator - Not Yet Implemented
RAG using existing body of pokemon moves as Prompt

Need to "sanity check" generations and pick the best
Multiple generations run through Ranking Classifier to pick the Best --> Ranking Classifier trained on existing pokemon

Dataset:
    scrape serebii for level and tm moves of pokemon in pokemon dataset

### Vision + Text LMM Pokedex + Stats Generator - Not Yet Implemented
LoRA FineTuned on The Pokemon Dataset
Tune the model to take an instruction prompt and the
generated character image to generate stats

Dummy version can be randomly generated, but the goal of this
project is to realistically generate the characters by spreading attributes effectively across a normal distribution

### Pipelines
Training Pipeline
Data Processing Pipeline

## References
- [Pokemon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon?resource=download)
- [Detailed Analysis and Model Training](https://www.kaggle.com/code/shobhit043/detailed-analysis-and-model-training-98-acc)<br>
- [Nickname Generation with Recurrent Neural Networks - Medium](https://medium.com/data-science-and-machine-learning-at-pluralsight/nickname-generation-with-recurrent-neural-networks-with-pytorch-6fa53de7f289)
- [CharLSTM Gist](https://gist.github.com/jrwalk/9ceee8707f01a324e72fbbe3ebc37e51#file-char_lstm-py)
- [Generic PyTorch Text Generator w/ Character LSTM](https://www.youtube.com/watch?v=WujVlF_6h5A)
- [PyTorch CharRNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [Apache Airflow Pipeline](https://towardsdatascience.com/10-minutes-to-building-a-machine-learning-pipeline-with-apache-airflow-53cd09268977)