#!/usr/bin/python
# -*- coding: latin-1 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# The jumping off point for this code was 
# https://keras.io/examples/nlp/text_classification_with_transformer/
# from https://github.com/keras-team/keras-io/

# I added dynamic linear combinations, Rezero, FC Rezero, Prenorm ...


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output



class Mega_Block(layers.Layer):
    def __init__(self, maxlen, embed_dim, num_heads, ff_dim, number, rate = 0.1):
        super(Mega_Block, self).__init__()

        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate = rate)

        self.concatenation_block =  Concatenation(maxlen,embed_dim,number-1)
        
        self.linear_combination_block = LinearCombination(number)
        

    def call(self, c, l):

        x = self.transformer_block(l)
        c2 = self.concatenation_block(c,x)
        l2 = self.linear_combination_block(c2)
        
        return c2,l2




class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class RezeroTransformerBlock(layers.Layer):
    """
    The rezero transformer block doesn't require layer normalisation
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.0):
        super(RezeroTransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.alpha = tf.Variable( initial_value=0.0, dtype="float32",  trainable=True)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + self.alpha * attn_output
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + self.alpha * ffn_output



# Da wir bereits layernorm1 haben, macht es vermutlich keinen Unterschied ... 
class PreNorm_TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm0 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        norm_input = self.layernorm1(inputs)
        attn_output = self.att(norm_inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Weighted Linear Combination of Output as Layerinput
class LinearCombination(layers.Layer):
    def __init__(self, number):
        super(LinearCombination, self).__init__()
        w_init = tf.random_normal_initializer(mean=1.0/float(number), stddev=0.00001, seed=None)  # Im Grunde einfach gleich gewichtet. 
        self.w = tf.Variable(  initial_value=w_init( shape=[number], dtype="float32" ),  trainable=True)

    def call(self, x):
        output_tensor = tf.tensordot(self.w,x,axes = [[0], [1]])

        return output_tensor

class Concatenation(layers.Layer):
    def __init__(self, maxlen, embed_dim, number):
        super(Concatenation, self).__init__()
        self.number = number
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.concat= layers.Concatenate(axis=1)

    def call(self, c, x):
        output = self.concat([tf.reshape(c,(-1,self.number,self.maxlen,self.embed_dim)),tf.reshape(x,(-1,1,self.maxlen,self.embed_dim))])
        return output





# Stacking Blocks 
class Multi_Block(layers.Layer):
    def __init__(self, maxlen, embed_dim, num_heads, ff_dim, number, rate = 0.0):
        super(Multi_Block, self).__init__()
        self.number = number
        self.mega_blocks = [Mega_Block(maxlen, embed_dim, num_heads, ff_dim, i+2 , rate = rate) for i in range(number)]
        

    def call(self, c,l):
        for i in range(self.number):
            c,l  = self.mega_blocks[i](c,l)
        return c,l





# rezero transformer model
class Rezero_Transformer(layers.Layer):
    def __init__(self, maxlen, embed_dim, num_heads, ff_dim, number_of_layers, rate = 0.0):
        super(Rezero_Transformer, self).__init__()
        self.number_of_layers = number_of_layers
        self.rezero_blocks = [ RezeroTransformerBlock( embed_dim, num_heads, ff_dim, rate = rate) for i in range(self.number_of_layers) ]
        #self.alphas = [ tf.Variable(  initial_value=0.0, dtype="float32",  trainable=True) for i in range(self.number_of_layers) ]

    def call(self, x):
        for i in range(self.number_of_layers):
            #x  = x + self.alphas[i] * self.rezero_blocks[i](x)
            x = self.rezero_blocks[i](x)
        return x




# rezero fully connected model
class Rezero_Fully_Connected(layers.Layer):
    def __init__(self, dimensions):
        super(Rezero_Fully_Connected, self).__init__()
        self.dimensions = dimensions
        self.Dense_Layers = [ keras.layers.Dense(dim, activation='relu') for dim in dimensions ]
        self.alphas = [ tf.Variable(  initial_value=0.0, dtype="float32",  trainable=True) for i in range(len(dimensions)) ]

    def call(self, x):
        for i in range(len(self.dimensions)):
            x  = x + self.alphas[i] * self.Dense_Layers[i](x)
        return x



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions








