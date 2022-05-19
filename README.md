# SpaceInvadersQ
A project based on Chloe Wang's Deep Q space invaders code, presented in a slightly different format. 

Check out the original code: https://github.com/nicknochnack/KerasRL-OpenAI-Atari-SpaceInvadersv0
It has a jupyter notebook giving a much more detailed walkthrough of the process, the code presented here is meant to run directly on terminal without much tweaking.

## Requirements
You'll need tensorflow, numpy, Keras-RL, and gym, plus their dependencies. This was tested with gym v0.21, 
some versions are picky about the way the code is implemented. You may need to modify Keras-RL, in the callbacks.py
file, by commenting out line 360. This is no longer needed with the version of gym we used, and it will
cause data.py to error out when it renders. You don't need to do this if you only intend to train.

We use the 1983 version of Space Invaders for our environment, which gym may be able to download if you accept the license. 
Otherwise, you may need to find out how to acquire the ROM (the popular pack you'll find by searching around no longer
includes the version retro/gym recognizes).

## Trained Models
If you want to see a model (albeit a very basic one) try to play, you can skip the training and
download the model we trained here: https://drive.google.com/file/d/1JimAGn-1g93R5pLwhqu5rddsE5XZP3IS/view?usp=sharing

This model was trained with 50,000 iterations. 
