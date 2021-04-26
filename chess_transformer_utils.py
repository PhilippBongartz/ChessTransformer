#!/usr/bin/python
# -*- coding: latin-1 -*-

import chess

import chess.pgn

import numpy as np

import random





# all possible pairs of starting square and target square

column_numbers = {
    'a':1,
    'b':2,
    'c':3,
    'd':4,
    'e':5,
    'f':6,
    'g':7,
    'h':8
}

columns = 'abcdefgh'
rows = '12345678'
squares = []
for c in columns:
    for r in rows:
        squares.append(''.join([c,r]))
        
square_pairs = []
for sq1 in squares:
    for sq2 in squares:
        square_pairs.append(''.join([sq1,sq2]))

index2moves = []  # ['0-0','0-0-0'] Rochade ist e1g1 etc.
for pair in square_pairs:
    if pair[0:2]!=pair[2:]: 
        if pair[0] == pair[2]: # gerade hoch
            index2moves.append(pair)
        if pair[1] == pair[3]:  # gerade seitwärts
            index2moves.append(pair)
        if abs(int(pair[1])-int(pair[3])) == abs(column_numbers[pair[0]]-column_numbers[pair[2]]): #diagonal
            index2moves.append(pair)
        if abs(int(pair[1])-int(pair[3])) == 1 and abs(column_numbers[pair[0]]-column_numbers[pair[2]])==2: # Springer
            index2moves.append(pair)
        if abs(int(pair[1])-int(pair[3])) == 2 and abs(column_numbers[pair[0]]-column_numbers[pair[2]])==1: # Springer
            index2moves.append(pair)
        
        if (pair[3]=='8' and pair[1]=='7') or (pair[3]=='1' and pair[1]=='2'): # Umwandlungen
            if abs(column_numbers[pair[0]]-column_numbers[pair[2]])<=1:
                index2moves.append(pair+'q')
                index2moves.append(pair+'n')
                index2moves.append(pair+'b')
                index2moves.append(pair+'r')

index2moves = list(set(index2moves))
index2moves = sorted(index2moves)

move2label = {}
for i,move in enumerate(index2moves + ['0000']):  # + Nullmove
    move2label[move] = i

# Hier die pure-transformer 64*64 output version: In der Reihenfolge der Token, d.h. auch der FEN.
# output[startsquareindex][zielsquareindex] 
pure_square = {}
for row in range(8,0,-1):
    for column in 'abcdefgh':
        pure_square[column+str(row)] = len(pure_square)


# Tokenization of a chess position

token2piece = '0KkQqBbNnRrPp'

fen2token_dict = {
    'K':np.array([1]),
    'k':np.array([2]),
    'Q':np.array([3]),
    'q':np.array([4]),
    'B':np.array([5]),
    'b':np.array([6]),
    'N':np.array([7]),
    'n':np.array([8]),
    'R':np.array([9]),
    'r':np.array([10]),
    'P':np.array([11]),
    'p':np.array([12]),
    '1':np.array([0]),
    '2':np.array([0,0]),
    '3':np.array([0,0,0]),
    '4':np.array([0,0,0,0]),
    '5':np.array([0,0,0,0,0]),
    '6':np.array([0,0,0,0,0,0]),
    '7':np.array([0,0,0,0,0,0,0]),
    '8':np.array([0,0,0,0,0,0,0,0]),
    '/':np.array([]),
}

castling2token = {
    'KQkq':np.array([13,14,15,16]),
    'Qkq':np.array([0,14,15,16]),
    'Kkq':np.array([13,0,15,16]),
    'KQq':np.array([13,14,0,16]),
    'KQk':np.array([13,14,15,0]),
    'KQ':np.array([13,14,0,0]),
    'Kk':np.array([14,0,15,0]),
    'Kq':np.array([13,0,0,16]),
    'Qk':np.array([0,14,15,0]),
    'Qq':np.array([0,14,0,16]),
    'kq':np.array([0,0,15,16]),
    'K':np.array([13,0,0,0]),
    'Q':np.array([0,14,0,0]),
    'k':np.array([0,0,15,0]),
    'q':np.array([0,0,0,16]),
    '-':np.array([0,0,0,0]),
}

column2token = {
    'a':np.array([17]),
    'b':np.array([18]),
    'c':np.array([19]),
    'd':np.array([20]),
    'e':np.array([21]),
    'f':np.array([22]),
    'g':np.array([23]),
    'h':np.array([24]),
    '-':np.array([0])
}

color2token = {
    'w':np.array([25]),
    'b':np.array([26])
}

def elo2token(elo):
    if elo == -1: # no elo
        token = 27
    elif elo < 1500:
        token = 28
    elif elo>=2700:
        token = 39
    else:
        token = 28 + (elo-1500)//100
    return np.array([token])


    
def fen2token(fen, white_elo = -1, black_elo = -1, move_list = [], elo = True):
    token_listen = []
    liste1 = fen.split()

    for l in liste1[0]:
        #print(l,fen2token_dict[l])
        token_listen.append(fen2token_dict[l])
    
    token_listen.append(color2token[liste1[1]]) # Wer am Zug ist
    
    token_listen.append(castling2token[liste1[2]]) # Rochaderechte
    
    token_listen.append(column2token[liste1[3][0]])  # en passant column

    if elo:
        token_listen.append(elo2token(white_elo))
        token_listen.append(elo2token(black_elo))
    
    if move_list:
        for move in move_list:
            token_listen.append(np.array([move2label[move]+40]))

    # Außerdem gibt es noch Halbzüge seit Bauernzug/Schlagzug und Zug der Partie. Die lassen wir weg.
    
    tokens = np.concatenate(token_listen)
    tokens = tokens.reshape((1,-1))
    return tokens

fen2vector_dict = {
    'K':np.array([1,0,0,0,0,0,0,0,0,0,0,0]),
    'k':np.array([0,1,0,0,0,0,0,0,0,0,0,0]),
    'Q':np.array([0,0,1,0,0,0,0,0,0,0,0,0]),
    'q':np.array([0,0,0,1,0,0,0,0,0,0,0,0]),
    'B':np.array([0,0,0,0,1,0,0,0,0,0,0,0]),
    'b':np.array([0,0,0,0,0,1,0,0,0,0,0,0]),
    'N':np.array([0,0,0,0,0,0,1,0,0,0,0,0]),
    'n':np.array([0,0,0,0,0,0,0,1,0,0,0,0]),
    'R':np.array([0,0,0,0,0,0,0,0,1,0,0,0]),
    'r':np.array([0,0,0,0,0,0,0,0,0,1,0,0]),
    'P':np.array([0,0,0,0,0,0,0,0,0,0,1,0]),
    'p':np.array([0,0,0,0,0,0,0,0,0,0,0,1]),
    '1':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*1),
    '2':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*2),
    '3':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*3),
    '4':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*4),
    '5':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*5),
    '6':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*6),
    '7':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*7),
    '8':np.array([0,0,0,0,0,0,0,0,0,0,0,0]*8),
    '/':np.array([]),
}

castling2vector = {
    'KQkq':np.array([1,1,1,1]),
    'Qkq':np.array([0,1,1,1]),
    'Kkq':np.array([1,0,1,1]),
    'KQq':np.array([1,1,0,1]),
    'KQk':np.array([1,1,1,0]),
    'KQ':np.array([1,1,0,0]),
    'Kk':np.array([1,0,1,0]),
    'Kq':np.array([1,0,0,1]),
    'Qk':np.array([0,1,1,0]),
    'Qq':np.array([0,1,0,1]),
    'kq':np.array([0,0,1,1]),
    'K':np.array([1,0,0,0]),
    'Q':np.array([0,1,0,0]),
    'k':np.array([0,0,1,0]),
    'q':np.array([0,0,0,1]),
    '-':np.array([0,0,0,0]),
}

column2vector = {
    'a':np.array([1,0,0,0,0,0,0,0]),
    'b':np.array([0,1,0,0,0,0,0,0]),
    'c':np.array([0,0,1,0,0,0,0,0]),
    'd':np.array([0,0,0,1,0,0,0,0]),
    'e':np.array([0,0,0,0,1,0,0,0]),
    'f':np.array([0,0,0,0,0,1,0,0]),
    'g':np.array([0,0,0,0,0,0,1,0]),
    'h':np.array([0,0,0,0,0,0,0,1]),
    '-':np.array([0,0,0,0,0,0,0,0])
}

color2vector = {
    'w':np.array([1,0]),
    'b':np.array([0,1])
}

def fen2vector(fen):
    vector_listen = []
    liste1 = fen.split()

    for l in liste1[0]:
        #print(l,fen2token_dict[l])
        vector_listen.append(fen2vector_dict[l])
    
    vector_listen.append(color2vector[liste1[1]]) # Wer am Zug ist
    
    vector_listen.append(castling2vector[liste1[2]]) # Rochaderechte
    
    vector_listen.append(column2vector[liste1[3][0]])  # en passant column
    
    # Außerdem gibt es noch Halbzüge seit Bauernzug/Schlagzug und Zug der Partie. Die lassen wir weg.
    
    vector = np.concatenate(vector_listen)
    vector = vector.reshape((1,-1))   # batch of one
    return vector






# OLD VERSION:
# validation_steps wird als Parameter an fit übergeben == total_validation_samples / batchsize 
def validationdata(path, batchsize, bis_game_number, stepnumber=60000, target = 'legalmove', aws = False):
    """
    targets können 'legalmove' sein, d.h. startfeld-zielfeld paare die tatsächlich auftreten können.
    'outcome', d.h. sieg, niederlage, remis
    'squarepairs', d.h. alle combinationen von start und zielfeld TODO
    'startingsquare', d.h. nur das startfeld TODO
    """ 
    #if aws: 
    #    from smart_open import open

    outcome_dict = {
    '1-0':np.array(0),
    '0-1':np.array(1),
    '1/2-1/2':np.array(2),
    }

    with open(path,encoding='latin-1') as database:
        print(chess.__version__,(chess.__version__ == '0.23.11'))
        current_game = ''
        batch = []
        labels = []
        count = 0
        step_count = 0
        while current_game != None:
            if count >= bis_game_number: # eternal loop
                database.seek(0)
                count = 0
                step_count = 0
                
            if step_count >= stepnumber: # eternal loop
                database.seek(0)
                count=0
                step_count = 0

            current_game = chess.pgn.read_game(database)
            board = current_game.board()
            count+=1

            if chess.__version__ == '0.23.11':
                current_game_moves = [move for move in current_game.main_line()]
            else:
                current_game_moves = [move for move in current_game.mainline_moves()]

            use_game = True
            if target == 'outcome':
                headers = current_game.headers
                if 'Result' in headers:
                    if headers['Result'] in ['1/2-1/2','0-1','1-0']:
                        outcome = outcome_dict[headers['Result']]
                    else:
                        use_game = False
                else:
                    use_game = False

            if current_game_moves and '0000' not in current_game_moves and use_game:
                for move in current_game_moves:
                    #print(move)
                    fen = board.fen()
                    tokens = fen2token(fen)
                    batch.append(tokens)
                    if target == 'legalmove':
                        labels.append(move2label[move.uci()])
                    elif target == 'outcome':
                        labels.append(outcome)
                    elif target == 'squarepairs':
                        movestring = move.uci()
                        startfeld_index = pure_square[movestring[:2]]
                        zielfeld_index  = pure_square[movestring[2:4]]
                        label = np.zeros(64*64)
                        label = label.reshape((64,64))
                        label[startfeld_index,zielfeld_index] = 1.0
                        label = label.reshape((64*64))  
                        labels.append(label)

                    board.push(move)
                    step_count += 1
                    
                    if len(batch)==batchsize:
                        batch_tensor = np.concatenate(batch)
                        yield batch_tensor, np.array(labels)
                        batch = []
                        labels = []



# OLD VERSION:
def trainingsdata(path, fraction, batchsize, from_game_number = 0, bis_game_number = 100000000, target = 'legalmove', aws = False):
    """
    targets können 'legalmove' sein, d.h. startfeld-zielfeld paare die tatsächlich auftreten können.
    'outcome', d.h. sieg, niederlage, remis
    'squarepairs', d.h. alle combinationen von start und zielfeld TODO
    'startingsquare', d.h. nur das startfeld TODO
    """

    outcome_dict = {
    '1-0':np.array(0),
    '0-1':np.array(1),
    '1/2-1/2':np.array(2),
    }

    #if aws: 
    #    from smart_open import open

    with open(path,encoding='latin-1') as database:
        current_game = ''
        batch = []
        labels = []
        count = 0
        while True: 
            # erstmal das erste game suchen, ohne parsen:
            #header = chess.pgn.read_headers(database)

            ## Skimming geht nicht auf der Colab python chess-version
            #while count < from_game_number:
            #    count+=1
            #    header = chess.pgn.read_headers(database)
            
            while current_game != None:
                #if count%1000 == 0:
                #    print(count)
            
                current_game = chess.pgn.read_game(database)
                board = current_game.board()
                count+=1
            
                if from_game_number < count < bis_game_number:
                    if chess.__version__ == '0.23.11':
                        current_game_moves = [move for move in current_game.main_line()]
                    else:
                        current_game_moves = [move for move in current_game.mainline_moves()]
                    current_game_movestrings = [move.uci() for move in current_game_moves]

                    use_game = True
                    if target == 'outcome':
                        headers = current_game.headers
                        if 'Result' in headers:
                            if headers['Result'] in ['1/2-1/2','0-1','1-0']:
                                outcome = outcome_dict[headers['Result']]
                            else:
                                use_game = False
                        else:
                            use_game = False


                    if current_game_moves and '0000' not in current_game_movestrings and use_game:
                        for move in current_game_moves:
                            #print(move)
                            rant = random.random()
                            if rant < fraction:
                                #print("Game no:",count,rant)
                                #print(board.fen())
                                fen = board.fen()
                                tokens = fen2token(fen)
                                batch.append(tokens)
                                if target == 'legalmove':
                                    labels.append(move2label[move.uci()])
                                elif target == 'outcome':
                                    labels.append(outcome)
                                elif target == 'squarepairs':
                                    movestring = move.uci()
                                    startfeld_index = pure_square[movestring[:2]]
                                    zielfeld_index  = pure_square[movestring[2:4]]
                                    label = np.zeros(64*64)
                                    label = label.reshape((64,64))
                                    label[startfeld_index,zielfeld_index] = 1.0
                                    label = label.reshape((64*64))  
                                    labels.append(label)

                                if len(batch)==batchsize:
                                    batch_tensor = np.concatenate(batch)
                                    yield batch_tensor, np.array(labels)
                                    batch = []
                                    labels = []
                            board.push(move)
                            
                #header = chess.pgn.read_headers(database)
                        
                if count >= bis_game_number:
                    current_game = None
                     
            database.seek(0) # wieder von vorne
            count = 0











outcome_dict = {
    '1-0':np.array(0),
    '0-1':np.array(1),
    '1/2-1/2':np.array(2),
}

# Encapsulation für bessere Lesbarkeit
def game_can_be_used(target,headers):
    if target == 'outcome':
        if 'Result' in headers:
            if headers['Result'] in ['1/2-1/2','0-1','1-0']:
                return True
            else:
                return False
        else:
            return False
    return True
                

def move_strings(current_game):
    if chess.__version__ == '0.23.11':
        current_game_moves = [move for move in current_game.main_line()]
    else:
        current_game_moves = [move for move in current_game.mainline_moves()]
    current_game_movestrings = [move.uci() for move in current_game_moves]
    return current_game_movestrings,current_game_moves


def get_elos(headers):
    white_elo = -1
    black_elo = -1
    
    if 'BlackElo' in headers:
        try:
            black_elo = int(headers['BlackElo'])
        except:
            pass

    if 'WhiteElo' in headers:
        try:
            white_elo = int(headers['WhiteElo'])
        except:
            pass
    
    return white_elo, black_elo


# Current data generator
def data_generator(path, fraction, batchsize, pool_size = 1, from_game_number = 0, bis_game_number = 100000000, target = 'legalmove', elo = False, move_tokens = 0, validation = False):
    """
    targets können 'legalmove' sein, d.h. startfeld-zielfeld paare die tatsächlich auftreten können.
    'outcome', d.h. sieg, niederlage, remis
    'squarepairs', d.h. alle combinationen von start und zielfeld
    'startingsquare', d.h. nur das startfeld TODO
    """
    #if aws: 
    #    from smart_open import open
    
    sample_label_pool = []

    with open(path,encoding='latin-1') as database:
        current_game = ''
        count = 0
        while True: 
            while current_game != None:
                current_game = chess.pgn.read_game(database)
                if current_game == None:
                    break

                board = current_game.board()
                count+=1

                use_game = game_can_be_used(target,current_game.headers)
                if target == 'outcome' and use_game:
                    outcome = outcome_dict[current_game.headers['Result']]
            
                if not (from_game_number < count < bis_game_number):
                    use_game = False
                
                current_game_moves = []
                current_game_movestrings = []
                if use_game:
                    current_game_movestrings,current_game_moves = move_strings(current_game)
                    
                if not (current_game_moves and '0000' not in current_game_movestrings and use_game):
                    use_game = False
                
                if use_game:
                    move_list = []
                    for t in range(move_tokens):
                        move_list.append('0000')
                        
                    for move in current_game_moves:
                        rant = random.random()
                        move_list.append(move.uci())
                        if rant < fraction:

                            white_elo, black_elo = -1,-1
                            if elo:
                                white_elo, black_elo = get_elos(current_game.headers)
                        
                            fen = board.fen()
                            tokens = fen2token(fen, white_elo=white_elo, black_elo=black_elo, move_list=move_list[-1*(move_tokens+1):-1], elo=elo )
                            
                            
                            if target == 'legalmove':
                                label = move2label[move.uci()]
                            elif target == 'outcome':
                                label = outcome
                            elif target == 'squarepairs':
                                movestring = move.uci()
                                startfeld_index = pure_square[movestring[:2]]
                                zielfeld_index  = pure_square[movestring[2:4]]
                                label = np.zeros(64*64)
                                label = label.reshape((64,64))
                                label[startfeld_index,zielfeld_index] = 1.0
                                label = label.reshape((64*64))  
                                    
                            sample_label_pool.append((tokens,label))
                                
                            if len(sample_label_pool)==batchsize * pool_size:
                                random.shuffle(sample_label_pool)
                                batch  = [t for (t,l) in sample_label_pool[:batchsize]]
                                labels = [l for (t,l) in sample_label_pool[:batchsize]]
                                batch_tensor = np.concatenate(batch)
                                yield batch_tensor, np.array(labels)
                                sample_label_pool = sample_label_pool[batchsize:] # besser in place
                            
                        board.push(move)

                if count >= bis_game_number:
                    current_game = None
                    if validation:
                        return
                     
            database.seek(0) # wieder von vorne
            current_game = ''
            count = 0  
