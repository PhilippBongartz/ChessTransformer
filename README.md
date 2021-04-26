# ChessTransformer
Code for a project to predict human chess moves with the Deep Learning architecture "Transformer". 

transformer.py contains the keras/tf code. It builds on https://keras.io/examples/nlp/text_classification_with_transformer/
which is why the Apache license is necessary. But this license allows you to do anything except sueing me. 

chess_transformer_utils.py is code for the data generator that is used to train the model. It uses python-chess. 
The code accomodates both the current python-chess version as well as the older version on Colab. 

Chesstransformer.ipynb is the Colab notebook that can be used with the above code to train a model to predict chess moves.
https://colab.research.google.com/drive/1QBDbHNbB_Gm7ZNYOQYiVTANoZ4QNdNBQ?usp=sharing
It requires you to have a pgn chess database on you Google drive. In this notebook 82 tokens are put into the model. 
70 encode the board position (64 squares + metainformation available in FEN), 10 encode the last moves, 2 encode elo. 
If you want to input only the board position the number of tokens and the size of the embeddings can be reduced accordingly.

PlayATransformer contains an older model and a notebook to play against that model.

chesstransformer_model.zip contains my best model with >51% accuracy on tournament games of human players with Elo ratings. 
The Elo ratings are not required. 
