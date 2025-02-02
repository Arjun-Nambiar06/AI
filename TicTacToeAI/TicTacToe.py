import pygame
import sys
import time
import math
import copy

X = "X"
O = "O"
EMPTY = None
Size = 3


def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    turn=0
    for i in board:
        for j in i:
            if j!=EMPTY:
                turn+=1
    if turn%2==0:
        return O
    else:
        return X


def actions(board):
    poss_move=[]
    for i in range(Size):
        for j in range(Size):
            if board[i][j]==EMPTY:
                poss_move.append((i,j))
    return poss_move


def result(board, action):
    (i,j) = action
    b_copy = copy.deepcopy(board)
    if b_copy[i][j] != EMPTY:
        raise Exception("Invalid Move.")
    elif player(b_copy) == O:
        b_copy[i][j] = O
    else:
        b_copy[i][j] = X
    return b_copy


def winner(board):
    for i in range(Size):
        if board[i][0] != EMPTY and all(board[i][j] == board[i][0] for j in range(Size)):
            return board[i][0]
        if board[0][i] != EMPTY and all(board[j][i] == board[0][i] for j in range(Size)):
            return board[0][i]
    if board[0][0] != EMPTY and all(board[i][i] == board[0][0] for i in range(Size)):
        return board[0][0]
    if board[0][Size-1] != EMPTY and all(board[i][Size-i-1] == board[0][Size-1] for i in range(Size)):
        return board[0][Size-1]
    return None

def terminal(board):
    if (winner(board) is not None) or (all(board[i][j] != EMPTY for j in range(Size) for i in range(Size))):
        return True
    else:
        return False

def utility(board):
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

def minimax(board,act_ret=True):
    if terminal(board):
        return utility(board) if not act_ret else None
    if player(board) == O:
        v_best = math.inf
        act_best = None
        for action in actions(board):
            v = minimax(result(board, action),act_ret=False)
            if v < v_best:
                v_best = v
                act_best = action
        return act_best if act_ret else v_best
    else:
        v_best = -(math.inf)
        act_best = None
        for action in actions(board):
            v = minimax(result(board, action),act_ret=False)
            if v > v_best:
                v_best = v
                act_best = action
        return act_best if act_ret else v_best

pygame.init()
size = width, height = 600, 400

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

screen = pygame.display.set_mode(size)

mediumFont = pygame.font.Font("AldotheApache.ttf", 28)
largeFont = pygame.font.Font("AldotheApache.ttf", 40)
moveFont = pygame.font.Font("AldotheApache.ttf", 60)

user = None
board = initial_state()
ai_turn = False

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(black)

    # Let user choose a player.
    if user is None:

        # Draw title
        title = largeFont.render("Play Tic-Tac-Toe", True, white)
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 50)
        screen.blit(title, titleRect)

        # Draw buttons
        playXButton = pygame.Rect((width / 8), (height / 2), width / 4, 50)
        playX = mediumFont.render("Play as X", True, black)
        playXRect = playX.get_rect()
        playXRect.center = playXButton.center
        pygame.draw.rect(screen, white, playXButton)
        screen.blit(playX, playXRect)

        playOButton = pygame.Rect(5 * (width / 8), (height / 2), width / 4, 50)
        playO = mediumFont.render("Play as O", True, black)
        playORect = playO.get_rect()
        playORect.center = playOButton.center
        pygame.draw.rect(screen, white, playOButton)
        screen.blit(playO, playORect)

        # Check if button is clicked
        click, _, _ = pygame.mouse.get_pressed()
        if click == 1:
            mouse = pygame.mouse.get_pos()
            if playXButton.collidepoint(mouse):
                time.sleep(0.2)
                user = X
            elif playOButton.collidepoint(mouse):
                time.sleep(0.2)
                user = O

    else:

        # Draw game board
        tile_size = 80
        tile_origin = (width / 2 - (1.5 * tile_size),
                       height / 2 - (1.5 * tile_size))
        tiles = []
        for i in range(3):
            row = []
            for j in range(3):
                rect = pygame.Rect(
                    tile_origin[0] + j * tile_size,
                    tile_origin[1] + i * tile_size,
                    tile_size, tile_size
                )
                pygame.draw.rect(screen, white, rect, 3)

                if board[i][j] != EMPTY:
                    move = moveFont.render(board[i][j], True, white)
                    moveRect = move.get_rect()
                    moveRect.center = rect.center
                    screen.blit(move, moveRect)
                row.append(rect)
            tiles.append(row)

        game_over = terminal(board)
        player_curr = player(board)

        # Show title
        if game_over:
            Winner = winner(board)
            if Winner is None:
                title = f"Game Over: Tie."
            else:
                title = f"Game Over: {Winner} wins."
        elif user == player_curr:
            title = f"Play as {user}"
        else:
            title = f"AI is thinking..."
        title = largeFont.render(title, True, white)
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 30)
        screen.blit(title, titleRect)

        # Check for AI move
        if user != player_curr and not game_over:
            if ai_turn:
                time.sleep(0.5)
                move = minimax(board)
                board = result(board, move)
                ai_turn = False
            else:
                ai_turn = True

        # Check for a user move
        click, _, _ = pygame.mouse.get_pressed()
        if click == 1 and user == player_curr and not game_over:
            mouse = pygame.mouse.get_pos()
            for i in range(3):
                for j in range(3):
                    if (board[i][j] == EMPTY and tiles[i][j].collidepoint(mouse)):
                        board = result(board, (i, j))

        if game_over:
            againButton = pygame.Rect(width / 3, height - 65, width / 3, 50)
            again = mediumFont.render("Play Again", True, black)
            againRect = again.get_rect()
            againRect.center = againButton.center
            pygame.draw.rect(screen, white, againButton)
            screen.blit(again, againRect)
            click, _, _ = pygame.mouse.get_pressed()
            if click == 1:
                mouse = pygame.mouse.get_pos()
                if againButton.collidepoint(mouse):
                    time.sleep(0.2)
                    user = None
                    board = initial_state()
                    ai_turn = False

    pygame.display.flip()
