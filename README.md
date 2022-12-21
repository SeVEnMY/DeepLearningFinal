# Playing Modified Flappy Bird Using Reinforcement Learning

This repository contains the code and documentation for a project on using deep reinforcement learning to play a modified version of the popular mobile game Flappy Bird. The modifications to the game include the addition of fireballs that fly across the screen and the widening of the gap between the pipes.

The Flappy Bird Pygame file used in this project was adapted from the repository "Flappy-bird-deep-Q-learning-pytorch" (https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch). We would like to acknowledge and thank the author for sharing this resource.

## Requirements

To run the code in this repository, you will need the following software and libraries installed on your machine:

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- OpenCV 4.0 or higher
- NumPy 1.19 or higher
- PyTorch 1.7 or higher
- TensorBoardX 2.0 or higher

## Training and Testing the Model

To train the model, open the `RL_flappy_bird.ipynb` notebook in a Jupyter environment and run all the cells before the subtitle "Test Procedure:". This will train the model and save it to the `models/` directory.

To test the model, run the cells after the subtitle "Test Procedure:". This will load the trained model from the `models/` directory and run it on the modified Flappy Bird game.

## File Structure

The repository includes the following files and directories:

- `RL_flappy_bird.ipynb`: A Jupyter notebook containing the code for training and testing the model.
- `results/`: A directory containing the trained model and any intermediate models generated during training.
- `src/`: Contains the flappy bird Pygame file.
- `assets/`: Contains assets for the flappy bird Pygame.
