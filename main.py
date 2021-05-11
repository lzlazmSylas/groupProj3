
import game

if __name__ == '__main__':
    playahs = []

    # playahs.append(game.RandomPlayer_Test("ran1"))
    # playahs.append(game.MCRLPlayer("AI"))
    # playahs.append(game.RandomPlayer_Test("ran2"))
    # Game = game.Game_Train(playahs)
    # Game.train()

    playahs.append(game.RandomPlayer("ran1"))
    playahs.append(game.MCRLPlayer("AI1", weight=  [0.0025237184329079624, -0.019729059122510512, -0.003707391469921135, 0.4909432195328273, 0.08298083095432464, 0.17603262527389368, 0.1999028951468271, 0.2644811048497763, 0.2437574651159098],
                                              b=2.419960437958876,epsilon=0.2))

    playahs.append(game.RandomPlayer("ran2"))

    Game = game.Game(playahs)
    Game.play()

