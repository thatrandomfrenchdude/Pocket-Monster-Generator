import os

from src.dataset import PokemonNameDataset
from src.name_generator import NameGenerator

def main():
    # vars
    model_path = 'pocket_monster_model.pt'
    train_new_model = True

    # datasets
    pokemon = PokemonNameDataset()
    # TODO: add other collections of names - digimon, etc

    # setup and initialize model class
    name_generator = NameGenerator(
        pokemon,
        epochs=8000,
        hidden_size=128,
        lr=0.01
    )
    # load or train the model, depending on if it exists
    print(os.path.exists(model_path))
    if os.path.exists(model_path):
        try:
            print(f" -> Loading model from {model_path}...")
            name_generator.load_model(model_path)
        except Exception as e:
            print(f" -> Error loading model: {e}")
            train_new_model = True
    
    if train_new_model:
        print(" -> Training new model...")
        name_generator.train()
        name_generator.save_model(model_path)

    # generate some names
    print(" -> Generating some names...")
    print(name_generator.generate())

if __name__ == '__main__':
    main()