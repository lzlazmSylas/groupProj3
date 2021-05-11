import random
import copy
import numpy as np
import abc
import time
from sklearn.linear_model import SGDRegressor


class Deck():
    def __init__(self):
        self.deck = []
        self.shuffle()

    def __str__(self):
        # Print the list without the brackets
        return str(self.deck).strip('[]')

    def shuffle(self):
        self.deck = []
        for suit in ['U', 'F', 'Z', 'T']:
            for i in range(15):
                self.deck.append((suit, i))
        random.shuffle(self.deck)

    def getCard(self):
        return self.deck.pop()


class Player(metaclass=abc.ABCMeta):  # This is an abstract base class.
    def __init__(self, name):
        self.name = name
        self.hand = []  # List of cards (tuples). I don't think this needs to be a class....
        self.score = 0
        self.zombie_count = 0

    def __repr__(self):  # If __str__ is not defined this will be used. Allows easy printing
        # of a list of these, e.g. "print(players)" below.
        return str(self.name) + ": " + str(self.hand) + "\n"

    @abc.abstractmethod
    def playCard(self, trick):
        pass

    def randomPlay(self, trick):
        random.shuffle(self.hand)
        # print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        if len(trick) != 0:
            # Figure out what was led and follow it if we can
            suit = trick[0][0]
            card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
            if card_idx != None:
                return self.hand.pop(card_idx)
        # If the trick is empty or if we can't follow suit, return anything
        return self.hand.pop()

    def directPlay(self, hand):
        self.hand.remove(hand)
        return hand

    def QfunctionPlay(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random()< epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def QfunctionPlay_rollout(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random() < epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def Q_play(self, trick, dummy_players, ai_id,dummy_p_idx, lead_player):
        allpossibleHands = self.allPossibleHands(trick)
        simulate_players_test = dummy_players
        trick_test = copy.deepcopy(trick)
        Q_list = np.zeros(len(allpossibleHands))
        feture_l = []
        for i in range(len(allpossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick_two(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    p_id=dummy_p_idx,
                                                                    ai_id= ai_id,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allpossibleHands[i],
                                                                      b= simulate_players_test[ai_id].b)
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print("q list",Q_list)
        # print(allpossibleHands)
        # print(max_index)
        # print(len(allpossibleHands))
        # print(feture_l[max_index])
        return allpossibleHands[max_index]

        # pass  # This is just a placeholder, remove when real code goes here
    def playOneTrick_two(self,dummy_players, trick,  p_id,ai_id, leader_id, hand_to_play,b):
        # for i in range(len(allPossibleHands)):
        #     state_list.append(State(dummy_players,trick, ai_id, leader_id))
        #     action_list.append(Action(dummy_players, trick, allPossibleHands[i],ai_id))
        state = State(dummy_players,trick, p_id, leader_id)
        action = Action(dummy_players, trick, hand_to_play,p_id)
        #     feature_list
        feature_list = np.zeros(9)
        feature_list[0] = state.score_difference
        # feature_list[1] = state.score_difference2
        feature_list[1] = state.score_difference2
        feature_list[2] = state.trick_point
        # feature_list[3] = state.z_total
        # feature_list[4] = state.t_total
        # feature_list[5] = state.u_total
        # feature_list[6] = state.f_total
        feature_list[3] = action.if_follow
        feature_list[4] = action.win_tag
        feature_list[5] = action.z_change
        feature_list[6] = action.t_change
        feature_list[7] = action.u_change
        feature_list[8] = action.f_change
        weight = dummy_players[ai_id].weight
        Q_value = np.matmul(weight, feature_list.T) + b
        return Q_value, feature_list, hand_to_play
    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand

        return allPossibleHands

class RandomPlayer(Player):  # Inherit from Player
    def __init__(self, name):
        super().__init__(name)
    def playCard(self, trick):
        random.shuffle(self.hand)
        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        if len(trick) != 0:
            # Figure out what was led and follow it if we can
            suit = trick[0][0]
            # print(self.name, ":", suit, "was led")
            # Get the first occurence of a matching suit in our hand
            # This 'next' thing below is a "generator expression"
            card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
            if card_idx != None:
                return self.hand.pop(card_idx)
        # If the trick is empty or if we can't follow suit, return anything
        return self.hand.pop()
    def QfunctionPlay_rollout(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random() < epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def Q_play(self, trick, dummy_players, ai_id,dummy_p_idx, lead_player):
        allpossibleHands = self.allPossibleHands(trick)
        simulate_players_test = dummy_players
        trick_test = copy.deepcopy(trick)
        Q_list = np.zeros(len(allpossibleHands))
        feture_l = []
        for i in range(len(allpossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick_two(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    p_id=dummy_p_idx,
                                                                    ai_id= ai_id,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allpossibleHands[i],
                                                                      b= simulate_players_test[ai_id].b)
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        return allpossibleHands[max_index]

        # pass  # This is just a placeholder, remove when real code goes here
    def playOneTrick_two(self,dummy_players, trick,  p_id,ai_id, leader_id, hand_to_play,b):
        # for i in range(len(allPossibleHands)):
        #     state_list.append(State(dummy_players,trick, ai_id, leader_id))
        #     action_list.append(Action(dummy_players, trick, allPossibleHands[i],ai_id))
        state = State(dummy_players,trick, p_id, leader_id)
        action = Action(dummy_players, trick, hand_to_play,p_id)
        #     feature_list
        feature_list = np.zeros(9)
        feature_list[0] = state.score_difference
        # feature_list[1] = state.score_difference2
        feature_list[1] = state.score_difference2
        feature_list[2] = state.trick_point
        # feature_list[3] = state.z_total
        # feature_list[4] = state.t_total
        # feature_list[5] = state.u_total
        # feature_list[6] = state.f_total
        feature_list[3] = action.if_follow
        feature_list[4] = action.win_tag
        feature_list[5] = action.z_change
        feature_list[6] = action.t_change
        feature_list[7] = action.u_change
        feature_list[8] = action.f_change
        weight = dummy_players[ai_id].weight
        Q_value = np.matmul(weight, feature_list.T) + b
        return Q_value, feature_list, hand_to_play
    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand

        return allPossibleHands

class GrabAndDuckPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def getindex(self, suit, int):  # int = 1, return maxindex, int = 0,return minindex
        min = 0
        max = 0
        card_idx = 0
        card_judge = 0
        if int == 1:
            while card_judge != None:
                max = self.hand[card_idx][1]
                card_judge = next((i for i, c in enumerate(self.hand)
                                   if i > card_idx and c[0] == suit and c[1] > max), None)
                if card_judge != None:
                    card_idx = card_judge
            return card_idx

        else:

            while card_judge != None:
                min = self.hand[0][1]
                card_judge = next((i for i, c in enumerate(self.hand)
                                   if i > card_idx and c[0] == suit and c[1] < min), None)
                if card_judge != None:
                    card_idx = card_judge

        return card_idx
    def QfunctionPlay_rollout(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random() < epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def Q_play(self, trick, dummy_players, ai_id,dummy_p_idx, lead_player):
        allpossibleHands = self.allPossibleHands(trick)
        simulate_players_test = dummy_players
        trick_test = copy.deepcopy(trick)
        Q_list = np.zeros(len(allpossibleHands))
        feture_l = []
        for i in range(len(allpossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick_two(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    p_id=dummy_p_idx,
                                                                    ai_id= ai_id,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allpossibleHands[i],
                                                                      b= simulate_players_test[ai_id].b)
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print("q list",Q_list)
        # print(allpossibleHands)
        # print(max_index)
        # print(len(allpossibleHands))
        # print(feture_l[max_index])
        return allpossibleHands[max_index]

        # pass  # This is just a placeholder, remove when real code goes here
    def playOneTrick_two(self,dummy_players, trick,  p_id,ai_id, leader_id, hand_to_play,b):
        # for i in range(len(allPossibleHands)):
        #     state_list.append(State(dummy_players,trick, ai_id, leader_id))
        #     action_list.append(Action(dummy_players, trick, allPossibleHands[i],ai_id))
        state = State(dummy_players,trick, p_id, leader_id)
        action = Action(dummy_players, trick, hand_to_play,p_id)
        #     feature_list
        feature_list = np.zeros(9)
        feature_list[0] = state.score_difference
        # feature_list[1] = state.score_difference2
        feature_list[1] = state.score_difference2
        feature_list[2] = state.trick_point
        # feature_list[3] = state.z_total
        # feature_list[4] = state.t_total
        # feature_list[5] = state.u_total
        # feature_list[6] = state.f_total
        feature_list[3] = action.if_follow
        feature_list[4] = action.win_tag
        feature_list[5] = action.z_change
        feature_list[6] = action.t_change
        feature_list[7] = action.u_change
        feature_list[8] = action.f_change
        weight = dummy_players[ai_id].weight
        Q_value = np.matmul(weight, feature_list.T) + b
        return Q_value, feature_list, hand_to_play
    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand

        return allPossibleHands

    def playCard(self, trick):
        random.shuffle(self.hand)
        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        min = 0
        max = 0
        card_judge = 0
        if len(trick) != 0:
            suit = trick[0][0]
            value = trick[0][1]
            if suit == 'U':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] > value), None)
                    if card_idx != None:
                        card_judge = card_idx
                        while card_judge != None:
                            min = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] > value and c[1] < min), None)
                            if card_judge != None:
                                card_idx = card_judge

                    else:
                        card_idx = self.getindex(suit, 0)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None)
                    if card_idx != None:
                        card_idx = self.getindex('T', 0)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None) != None:
                        card_idx = self.getindex('Z', 1)
                    else:
                        card_idx = self.getindex('F', 0)

                return self.hand.pop(card_idx)

            if suit == 'F':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] > value), None)
                    if card_idx != None:
                        while card_judge != None:
                            min = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] > value and c[1] < min), None)
                            if card_judge != None:
                                card_idx = card_judge
                    else:
                        card_idx = self.getindex(suit, 0)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None)
                    if card_idx != None:
                        card_idx = self.getindex('Z', 1)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None) != None:
                        card_idx = self.getindex('T', 0)
                    else:
                        card_idx = self.getindex('U', 0)

                return self.hand.pop(card_idx)

            if suit == 'Z':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] < value), None)
                    if card_idx != None:

                        while card_judge != None:
                            max = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] < value and c[1] > max), None)
                            if card_judge != None:
                                card_idx = card_judge

                    else:
                        card_idx = self.getindex(suit, 1)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None)
                    if card_idx != None:
                        card_idx = self.getindex('T', 0)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'F'), None) != None:
                        card_idx = self.getindex('F', 0)
                    else:
                        card_idx = self.getindex('U', 0)

                return self.hand.pop(card_idx)

            if suit == 'T':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] > value), None)
                    if card_idx != None:
                        while card_judge != None:
                            min = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] > value and c[1] < min), None)
                            if card_judge != None:
                                card_idx = card_judge

                    else:
                        card_idx = self.getindex(suit, 0)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None)
                    if card_idx != None:
                        card_idx = self.getindex('Z', 1)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'F'), None) != None:
                        card_idx = self.getindex('F', 0)
                    else:
                        card_idx = self.getindex('U', 0)

                return self.hand.pop(card_idx)

        else:
            card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None)
            if card_idx != None:
                card_idx = self.getindex('T', 0)
            elif next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None) != None:
                card_idx = self.getindex('Z', 0)
            elif next((i for i, c in enumerate(self.hand) if c[0] == 'F'), None) != None:
                card_idx = self.getindex('F', 1)
            else:
                card_idx = self.getindex('U', 1)
            return self.hand.pop(card_idx)

class RandomPlayer_Test(Player):  # Inherit from Player
    def __init__(self, name):
        super().__init__(name)

    def playCard(self, trick):
        random.shuffle(self.hand)
        # print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        if len(trick) != 0:
            # Figure out what was led and follow it if we can
            suit = trick[0][0]
            # print(self.name, ":", suit, "was led")
            # Get the first occurence of a matching suit in our hand
            # This 'next' thing below is a "generator expression"
            card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
            if card_idx != None:
                return self.hand.pop(card_idx)
        # If the trick is empty or if we can't follow suit, return anything
        return self.hand.pop()

class GrabAndDuckPlayer_Test(Player):
    def __init__(self, name):
        super().__init__(name)

    def getindex(self, suit, int):  # int = 1, return maxindex, int = 0,return minindex
        min = 0
        max = 0
        card_idx = 0
        card_judge = 0
        if int == 1:
            while card_judge != None:
                max = self.hand[card_idx][1]
                card_judge = next((i for i, c in enumerate(self.hand)
                                   if i > card_idx and c[0] == suit and c[1] > max), None)
                if card_judge != None:
                    card_idx = card_judge
            return card_idx

        else:

            while card_judge != None:
                min = self.hand[0][1]
                card_judge = next((i for i, c in enumerate(self.hand)
                                   if i > card_idx and c[0] == suit and c[1] < min), None)
                if card_judge != None:
                    card_idx = card_judge

        return card_idx

    def playCard(self, trick):
        random.shuffle(self.hand)
        # print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        min = 0
        max = 0
        card_judge = 0
        if len(trick) != 0:
            suit = trick[0][0]
            value = trick[0][1]
            if suit == 'U':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] > value), None)
                    if card_idx != None:
                        card_judge = card_idx
                        while card_judge != None:
                            min = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] > value and c[1] < min), None)
                            if card_judge != None:
                                card_idx = card_judge

                    else:
                        card_idx = self.getindex(suit, 0)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None)
                    if card_idx != None:
                        card_idx = self.getindex('T', 0)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None) != None:
                        card_idx = self.getindex('Z', 1)
                    else:
                        card_idx = self.getindex('F', 0)

                return self.hand.pop(card_idx)

            if suit == 'F':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] > value), None)
                    if card_idx != None:
                        while card_judge != None:
                            min = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] > value and c[1] < min), None)
                            if card_judge != None:
                                card_idx = card_judge
                    else:
                        card_idx = self.getindex(suit, 0)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None)
                    if card_idx != None:
                        card_idx = self.getindex('Z', 1)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None) != None:
                        card_idx = self.getindex('T', 0)
                    else:
                        card_idx = self.getindex('U', 0)

                return self.hand.pop(card_idx)

            if suit == 'Z':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] < value), None)
                    if card_idx != None:

                        while card_judge != None:
                            max = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] < value and c[1] > max), None)
                            if card_judge != None:
                                card_idx = card_judge

                    else:
                        card_idx = self.getindex(suit, 1)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None)
                    if card_idx != None:
                        card_idx = self.getindex('T', 0)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'F'), None) != None:
                        card_idx = self.getindex('F', 0)
                    else:
                        card_idx = self.getindex('U', 0)

                return self.hand.pop(card_idx)

            if suit == 'T':
                card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit), None)
                if card_idx != None:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == suit and c[1] > value), None)
                    if card_idx != None:
                        while card_judge != None:
                            min = self.hand[card_idx][1]
                            card_judge = next((i for i, c in enumerate(self.hand)
                                               if i > card_idx and c[0] == suit and c[1] > value and c[1] < min), None)
                            if card_judge != None:
                                card_idx = card_judge

                    else:
                        card_idx = self.getindex(suit, 0)
                else:
                    card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None)
                    if card_idx != None:
                        card_idx = self.getindex('Z', 1)
                    elif next((i for i, c in enumerate(self.hand) if c[0] == 'F'), None) != None:
                        card_idx = self.getindex('F', 0)
                    else:
                        card_idx = self.getindex('U', 0)

                return self.hand.pop(card_idx)

        else:
            card_idx = next((i for i, c in enumerate(self.hand) if c[0] == 'T'), None)
            if card_idx != None:
                card_idx = self.getindex('T', 0)
            elif next((i for i, c in enumerate(self.hand) if c[0] == 'Z'), None) != None:
                card_idx = self.getindex('Z', 0)
            elif next((i for i, c in enumerate(self.hand) if c[0] == 'F'), None) != None:
                card_idx = self.getindex('F', 1)
            else:
                card_idx = self.getindex('U', 1)
            return self.hand.pop(card_idx)


class RolloutPlayer(Player):
    def __init__(self, name, total_playerNum=3, time_limit=1):
        super().__init__(name)

        self.trick = None
        self.time_limit = time_limit
        self.dummy_players = None
        self.dummy_p_idx = None
        self.undealed = None
        self.lead_player = None
        self.simulate_players = None
        self.simulated_undealed = None
        self.unknownHands = None



    def playCard(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        allPossibleHands = self.allPossibleHands(trick)

        gain_list = list(np.zeros(len(allPossibleHands)))

        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)

        self.dummy_p_idx = dummy_p_idx
        self.dummy_players = dummy_players
        self.simulate_players = copy.deepcopy(dummy_players)
        self.undealed = undealed
        self.trick = trick
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        self.getAllUnknownHands()

        start_time = time.time()
        time_cost = 0
        while time_cost < self.time_limit:
            for i in range(0, len(allPossibleHands)):
                this_gain = self.oneRound(startingTrick=allPossibleHands[i])
                gain_list[i] += this_gain

            time_cost = time.time() - start_time

        max_index = gain_list.index(max(gain_list))
        hand_to_play = allPossibleHands[max_index]
        self.hand.remove(hand_to_play)
        return hand_to_play

        # pop the one with highest gain

        # pass  # This is just a placeholder, remove when real code goes here

    def oneRound(self, startingTrick):
        gain = 0
        self.create_simulation_players()
        oneRound = Game(self.simulate_players)
        oneRound.deck.deck = self.simulated_undealed
        scoreDifference = oneRound.playOneRound(trick=copy.copy(self.trick),
                                                lead_player=copy.copy(self.lead_player),
                                                trickstartingTrick=copy.copy(startingTrick))

        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                gain += (scoreDifference[self.dummy_p_idx] - scoreDifference[i])
        # return simulate_players
        return gain

    # create simulated players and undealed hand
    def create_simulation_players(self):
        unknownHands = copy.copy(self.unknownHands)
        random.shuffle(unknownHands)
        for i in range(0, len(self.dummy_players)):
            self.simulate_players[i].name = copy.copy(self.dummy_players[i].name)
            self.simulate_players[i].hand = copy.copy(self.dummy_players[i].hand)
            self.simulate_players[i].score = copy.copy(self.dummy_players[i].score)
            self.simulate_players[i].zombie_count = copy.copy(self.dummy_players[i].zombie_count)
            # test = copy.deepcopy(self.dummy_players)
        # self.simulate_players = test
        # for i in range(0, len(self.dummy_players)):
        j = 0
        while j < len(self.dummy_players):
            if j != self.dummy_p_idx:
                # length = len(self.dummy_players[i].hand)
                # a = unknownHands[0:length]
                # self.simulate_players[i].hand = a
                # unknownHands = unknownHands[length:]

                length = len(self.dummy_players[j].hand)
                self.simulate_players[j].hand.clear()
                count = 0
                while self.playerHasNo[j][unknownHands[0][0]] and length > 0:
                    count = count + 1
                    if count > len(unknownHands):
                        unknownHands = copy.copy(self.unknownHands)
                        break
                    unknownHands.append(unknownHands.pop(0))
                if count > len(unknownHands):
                    j = 0
                    continue
                a = unknownHands[0:length]
                self.simulate_players[j].hand = a
                unknownHands = unknownHands[length:]
                random.shuffle(unknownHands)
            j = j + 1

        self.simulated_undealed = unknownHands

    def getAllUnknownHands(self):
        unknownHands = []
        # all unknown cards
        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                unknownHands.append(self.dummy_players[i].hand)
        unknownHands.append(self.undealed)
        unknownHands = sum(unknownHands, [])

        # shuffle unknown hands
        random.shuffle(unknownHands)
        self.unknownHands = unknownHands

    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand
        return allPossibleHands
    def QfunctionPlay_rollout(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random() < epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def Q_play(self, trick, dummy_players, ai_id,dummy_p_idx, lead_player):
        allpossibleHands = self.allPossibleHands(trick)
        simulate_players_test = dummy_players
        trick_test = copy.deepcopy(trick)
        Q_list = np.zeros(len(allpossibleHands))
        feture_l = []
        for i in range(len(allpossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick_two(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    p_id=dummy_p_idx,
                                                                    ai_id= ai_id,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allpossibleHands[i],
                                                                      b= simulate_players_test[ai_id].b)
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print("q list",Q_list)
        # print(allpossibleHands)
        # print(max_index)
        # print(len(allpossibleHands))
        # print(feture_l[max_index])
        return allpossibleHands[max_index]
        # pass  # This is just a placeholder, remove when real code goes here
    def playOneTrick_two(self,dummy_players, trick,  p_id,ai_id, leader_id, hand_to_play,b):
        # for i in range(len(allPossibleHands)):
        #     state_list.append(State(dummy_players,trick, ai_id, leader_id))
        #     action_list.append(Action(dummy_players, trick, allPossibleHands[i],ai_id))
        state = State(dummy_players,trick, p_id, leader_id)
        action = Action(dummy_players, trick, hand_to_play,p_id)
        #     feature_list
        feature_list = np.zeros(9)
        feature_list[0] = state.score_difference
        # feature_list[1] = state.score_difference2
        feature_list[1] = state.score_difference2
        feature_list[2] = state.trick_point
        # feature_list[3] = state.z_total
        # feature_list[4] = state.t_total
        # feature_list[5] = state.u_total
        # feature_list[6] = state.f_total
        feature_list[3] = action.if_follow
        feature_list[4] = action.win_tag
        feature_list[5] = action.z_change
        feature_list[6] = action.t_change
        feature_list[7] = action.u_change
        feature_list[8] = action.f_change
        weight = dummy_players[ai_id].weight
        Q_value = np.matmul(weight, feature_list.T) + b
        return Q_value, feature_list, hand_to_play
class MCRLPlayer(Player):
    def __init__(self, name, total_playerNum=3, time_limit=1, weight = np.zeros(9),b=0, epsilon=0.1):
        super().__init__(name)
        self.trick = None
        self.time_limit = time_limit
        self.dummy_players = None
        self.dummy_p_idx = None
        self.undealed = None
        self.lead_player = None
        self.simulate_players = None
        self.simulate_players_test = None
        self.simulated_undealed = None
        self.unknownHands = None
        self.weight = weight
        self.win_score = []
        self.y_label = []
        self.b = b
        self.epsilon = epsilon

    def playCard(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        allPossibleHands = self.allPossibleHands(trick)
        reward_list = list(np.zeros(len(allPossibleHands)))
        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        # print("ai hand", self.hand)
        self.dummy_p_idx = dummy_p_idx
        self.dummy_players = dummy_players
        self.simulate_players = copy.deepcopy(dummy_players)
        self.undealed = undealed
        self.trick = trick
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        self.getAllUnknownHands()

        start_time = time.time()
        time_cost = 0
        while time_cost < self.time_limit:
            for i in range(0, len(allPossibleHands)):
                this_reward = self.oneRound(startingTrick=allPossibleHands[i])
                reward_list[i] += this_reward
            time_cost = time.time() - start_time
        max_index = reward_list.index(max(reward_list))
        hand_to_play = allPossibleHands[max_index]
        self.hand.remove(hand_to_play)
        return hand_to_play
    def QfunctionPlay(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random()< epsilon:
            print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
            return self.randomPlay(trick)
        else:
            print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def QfunctionPlay_rollout(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random() < epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def Q_play(self, trick, dummy_players, ai_id,dummy_p_idx, lead_player):
        allpossibleHands = self.allPossibleHands(trick)
        simulate_players_test = dummy_players
        trick_test = copy.deepcopy(trick)
        Q_list = np.zeros(len(allpossibleHands))
        feture_l = []
        for i in range(len(allpossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick_two(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    p_id=dummy_p_idx,
                                                                    ai_id= ai_id,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allpossibleHands[i],
                                                                      b= simulate_players_test[ai_id].b)
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        return allpossibleHands[max_index]

        # pass  # This is just a placeholder, remove when real code goes here
    def Q_playCard_test(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        # print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        # print("trick",trick)
        allPossibleHands = self.allPossibleHands(trick)
        # print(self.hand)
        # print(allPossibleHands)
        simulate_players_test = copy.deepcopy(dummy_players)
        trick_test = copy.deepcopy(trick)
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        Q_list = np.zeros(len(allPossibleHands))
        # print(allPossibleHands)
        feture_l = []
        for i in range(len(allPossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    ai_id=dummy_p_idx,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allPossibleHands[i])
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print(feture_l[max_index])
        features = feture_l[max_index]
        # print( allPossibleHands[max_index])
        self.hand.remove(allPossibleHands[max_index])
        return simulate_players_test[dummy_p_idx].hand[max_index], features
    # create simulated players and undealed hand
    def Q_playCard_test2(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        # print("trick",trick)
        allPossibleHands = self.allPossibleHands(trick)
        # print(self.hand)
        # print(allPossibleHands)
        simulate_players_test = copy.deepcopy(dummy_players)
        trick_test = copy.deepcopy(trick)
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        Q_list = np.zeros(len(allPossibleHands))
        # print(allPossibleHands)
        feture_l = []
        for i in range(len(allPossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    ai_id=dummy_p_idx,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allPossibleHands[i])
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print(feture_l[max_index])
        features = feture_l[max_index]
        # print( allPossibleHands[max_index])
        self.hand.remove(allPossibleHands[max_index])
        return simulate_players_test[dummy_p_idx].hand[max_index], features
    def create_simulation_players(self):
        unknownHands = copy.copy(self.unknownHands)
        random.shuffle(unknownHands)
        for i in range(0, len(self.dummy_players)):
            self.simulate_players[i].name = copy.copy(self.dummy_players[i].name)
            self.simulate_players[i].hand = copy.copy(self.dummy_players[i].hand)
            self.simulate_players[i].score = copy.copy(self.dummy_players[i].score)
            self.simulate_players[i].zombie_count = copy.copy(self.dummy_players[i].zombie_count)
            # test = copy.deepcopy(self.dummy_players)
        # self.simulate_players = test
        # for i in range(0, len(self.dummy_players)):
        j = 0
        while j < len(self.dummy_players):
            if j != self.dummy_p_idx:
                # length = len(self.dummy_players[i].hand)
                # a = unknownHands[0:length]
                # self.simulate_players[i].hand = a
                # unknownHands = unknownHands[length:]

                length = len(self.dummy_players[j].hand)
                self.simulate_players[j].hand.clear()
                count = 0
                while self.playerHasNo[j][unknownHands[0][0]] and length > 0:
                    count = count + 1
                    if count > len(unknownHands):
                        unknownHands = copy.copy(self.unknownHands)
                        break
                    unknownHands.append(unknownHands.pop(0))
                if count > len(unknownHands):
                    j = 0
                    continue
                a = unknownHands[0:length]
                self.simulate_players[j].hand = a
                unknownHands = unknownHands[length:]
                random.shuffle(unknownHands)
            j = j + 1

        self.simulated_undealed = unknownHands
    def getAllUnknownHands(self):
        unknownHands = []
        # all unknown cards
        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                unknownHands.append(self.dummy_players[i].hand)
        unknownHands.append(self.undealed)
        unknownHands = sum(unknownHands, [])

        # shuffle unknown hands
        random.shuffle(unknownHands)
        self.unknownHands = unknownHands

    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand

        return allPossibleHands

    def generate_random_socre(self, dummy_players):
        max = 200
        one_player = random.randint(0, 199)
        two_player = random.randint(0, max - one_player)
        three_player = random.randint(0,max - one_player - two_player )
        dummy_players[0].score = one_player
        dummy_players[1].score = two_player
        dummy_players[2].score = three_player
    # we need find q value for each possible hand in each trick
    def playOneTrick(self,dummy_players, trick,  ai_id, leader_id, hand_to_play):
        # for i in range(len(allPossibleHands)):
        #     state_list.append(State(dummy_players,trick, ai_id, leader_id))
        #     action_list.append(Action(dummy_players, trick, allPossibleHands[i],ai_id))
        state = State(dummy_players,trick, ai_id, leader_id)
        action = Action(dummy_players, trick, hand_to_play,ai_id)
        #     feature_list
        feature_list = np.zeros(9)
        feature_list[0] = state.score_difference
        # feature_list[1] = state.score_difference2
        feature_list[1] = state.score_difference2
        feature_list[2] = state.trick_point
        # feature_list[3] = state.z_total
        # feature_list[4] = state.t_total
        # feature_list[5] = state.u_total
        # feature_list[6] = state.f_total
        feature_list[3] = action.if_follow
        feature_list[4] = action.win_tag
        feature_list[5] = action.z_change
        feature_list[6] = action.t_change
        feature_list[7] = action.u_change
        feature_list[8] = action.f_change
        weight = dummy_players[ai_id].weight
        Q_value = np.matmul(weight, feature_list.T) + self.b
        return Q_value, feature_list, hand_to_play
    def oneRound(self, startingTrick):
        reward = 0
        self.create_simulation_players()
        oneRound = Game(self.simulate_players)
        oneRound.deck.deck = self.simulated_undealed
        scoreDifference = oneRound.playOneRound(trick=copy.copy(self.trick),
                                                lead_player=copy.copy(self.lead_player),
                                                trickstartingTrick=copy.copy(startingTrick))

        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                reward += (scoreDifference[self.dummy_p_idx] - scoreDifference[i])
        # return simulate_players
        return reward
class MCRLPlayer_No_Rollout(Player):
    def __init__(self, name, total_playerNum=3, time_limit=1, weight = np.ones(9),b=1, epsilon=0.1):
        super().__init__(name)
        self.trick = None
        self.time_limit = time_limit
        self.dummy_players = None
        self.dummy_p_idx = None
        self.undealed = None
        self.lead_player = None
        self.simulate_players = None
        self.simulate_players_test = None
        self.simulated_undealed = None
        self.unknownHands = None
        self.weight = weight
        self.win_score = []
        self.y_label = []
        self.b = b
        self.epsilon = epsilon

    def playCard(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        allPossibleHands = self.allPossibleHands(trick)
        reward_list = list(np.zeros(len(allPossibleHands)))
        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        # print("ai hand", self.hand)
        self.dummy_p_idx = dummy_p_idx
        self.dummy_players = dummy_players
        self.simulate_players = copy.deepcopy(dummy_players)
        self.undealed = undealed
        self.trick = trick
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        self.getAllUnknownHands()

        start_time = time.time()
        time_cost = 0
        while time_cost < self.time_limit:
            for i in range(0, len(allPossibleHands)):
                this_reward = self.oneRound(startingTrick=allPossibleHands[i])
                reward_list[i] += this_reward
            time_cost = time.time() - start_time
        max_index = reward_list.index(max(reward_list))
        hand_to_play = allPossibleHands[max_index]
        self.hand.remove(hand_to_play)
        return hand_to_play
    def QfunctionPlay(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random()< epsilon:
            print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
            return self.randomPlay(trick)
        else:
            print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def QfunctionPlay_rollout(self, trick, epsilon, dummy_players, ai_id,p_id, lead_player):
        if random.random() < epsilon:
            return self.randomPlay(trick)
        else:
            hand_to_play = self.Q_play(trick,dummy_players,ai_id,p_id,lead_player)
            self.hand.remove(hand_to_play)
            return hand_to_play

    def Q_play(self, trick, dummy_players, ai_id,dummy_p_idx, lead_player):
        allpossibleHands = self.allPossibleHands(trick)
        simulate_players_test = dummy_players
        trick_test = copy.deepcopy(trick)
        Q_list = np.zeros(len(allpossibleHands))
        feture_l = []
        for i in range(len(allpossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick_two(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    p_id=dummy_p_idx,
                                                                    ai_id= ai_id,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allpossibleHands[i],
                                                                      b= simulate_players_test[ai_id].b)
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        return allpossibleHands[max_index]

        # pass  # This is just a placeholder, remove when real code goes here
    def Q_playCard_test(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        # print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        # print("trick",trick)
        allPossibleHands = self.allPossibleHands(trick)
        # print(self.hand)
        # print(allPossibleHands)
        simulate_players_test = copy.deepcopy(dummy_players)
        trick_test = copy.deepcopy(trick)
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        Q_list = np.zeros(len(allPossibleHands))
        # print(allPossibleHands)
        feture_l = []
        for i in range(len(allPossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    ai_id=dummy_p_idx,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allPossibleHands[i])
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print(feture_l[max_index])
        features = feture_l[max_index]
        # print( allPossibleHands[max_index])
        self.hand.remove(allPossibleHands[max_index])
        return simulate_players_test[dummy_p_idx].hand[max_index], features
    # create simulated players and undealed hand
    def Q_playCard_test2(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):
        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        # print("trick",trick)
        allPossibleHands = self.allPossibleHands(trick)
        # print(self.hand)
        # print(allPossibleHands)
        simulate_players_test = copy.deepcopy(dummy_players)
        trick_test = copy.deepcopy(trick)
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        Q_list = np.zeros(len(allPossibleHands))
        # print(allPossibleHands)
        feture_l = []
        for i in range(len(allPossibleHands)):
            Q_list[i], feature_list, hand_to_play = self.playOneTrick(dummy_players= simulate_players_test,
                                                                    trick = trick_test,
                                                                    ai_id=dummy_p_idx,
                                                                    leader_id=lead_player,
                                                                    hand_to_play= allPossibleHands[i])
            # q value and features for that q
            feture_l.append(feature_list)
        max_index = np.argmax(Q_list)
        # print(feture_l[max_index])
        features = feture_l[max_index]
        # print( allPossibleHands[max_index])
        self.hand.remove(allPossibleHands[max_index])
        return simulate_players_test[dummy_p_idx].hand[max_index], features
    def create_simulation_players(self):
        unknownHands = copy.copy(self.unknownHands)
        random.shuffle(unknownHands)
        for i in range(0, len(self.dummy_players)):
            self.simulate_players[i].name = copy.copy(self.dummy_players[i].name)
            self.simulate_players[i].hand = copy.copy(self.dummy_players[i].hand)
            self.simulate_players[i].score = copy.copy(self.dummy_players[i].score)
            self.simulate_players[i].zombie_count = copy.copy(self.dummy_players[i].zombie_count)
            # test = copy.deepcopy(self.dummy_players)
        # self.simulate_players = test
        # for i in range(0, len(self.dummy_players)):
        j = 0
        while j < len(self.dummy_players):
            if j != self.dummy_p_idx:
                # length = len(self.dummy_players[i].hand)
                # a = unknownHands[0:length]
                # self.simulate_players[i].hand = a
                # unknownHands = unknownHands[length:]

                length = len(self.dummy_players[j].hand)
                self.simulate_players[j].hand.clear()
                count = 0
                while self.playerHasNo[j][unknownHands[0][0]] and length > 0:
                    count = count + 1
                    if count > len(unknownHands):
                        unknownHands = copy.copy(self.unknownHands)
                        break
                    unknownHands.append(unknownHands.pop(0))
                if count > len(unknownHands):
                    j = 0
                    continue
                a = unknownHands[0:length]
                self.simulate_players[j].hand = a
                unknownHands = unknownHands[length:]
                random.shuffle(unknownHands)
            j = j + 1

        self.simulated_undealed = unknownHands
    def getAllUnknownHands(self):
        unknownHands = []
        # all unknown cards
        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                unknownHands.append(self.dummy_players[i].hand)
        unknownHands.append(self.undealed)
        unknownHands = sum(unknownHands, [])

        # shuffle unknown hands
        random.shuffle(unknownHands)
        self.unknownHands = unknownHands

    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand

        return allPossibleHands

    def generate_random_socre(self, dummy_players):
        max = 200
        one_player = random.randint(0, 199)
        two_player = random.randint(0, max - one_player)
        three_player = random.randint(0,max - one_player - two_player )
        dummy_players[0].score = one_player
        dummy_players[1].score = two_player
        dummy_players[2].score = three_player
    # we need find q value for each possible hand in each trick
    def playOneTrick(self,dummy_players, trick,  ai_id, leader_id, hand_to_play):
        # for i in range(len(allPossibleHands)):
        #     state_list.append(State(dummy_players,trick, ai_id, leader_id))
        #     action_list.append(Action(dummy_players, trick, allPossibleHands[i],ai_id))
        state = State(dummy_players,trick, ai_id, leader_id)
        action = Action(dummy_players, trick, hand_to_play,ai_id)
        #     feature_list
        feature_list = np.zeros(9)
        feature_list[0] = state.score_difference
        # feature_list[1] = state.score_difference2
        feature_list[1] = state.score_difference2
        feature_list[2] = state.trick_point
        # feature_list[3] = state.z_total
        # feature_list[4] = state.t_total
        # feature_list[5] = state.u_total
        # feature_list[6] = state.f_total
        feature_list[3] = action.if_follow
        feature_list[4] = action.win_tag
        feature_list[5] = action.z_change
        feature_list[6] = action.t_change
        feature_list[7] = action.u_change
        feature_list[8] = action.f_change
        weight = dummy_players[ai_id].weight
        Q_value = np.matmul(weight, feature_list.T) + self.b
        return Q_value, feature_list, hand_to_play
    def oneRound(self, startingTrick):
        reward = 0
        self.create_simulation_players()
        oneRound = Game(self.simulate_players)
        oneRound.deck.deck = self.simulated_undealed
        scoreDifference = oneRound.playOneRound(trick=copy.copy(self.trick),
                                                lead_player=copy.copy(self.lead_player),
                                                trickstartingTrick=copy.copy(startingTrick))

        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                reward += (scoreDifference[self.dummy_p_idx] - scoreDifference[i])
        # return simulate_players
        return reward
class MctsPlayer(Player):
    def __init__(self, name, total_playerNum=3, time_limit=1):
        super().__init__(name)

        self.trick = None
        self.time_limit = time_limit
        self.dummy_players = None
        self.dummy_p_idx = None
        self.undealed = None
        self.lead_player = None

        self.simulate_players = None
        self.simulated_undealed = None

        self.unknownHands = None

    def oneRound(self, startingTrick):
        gain = 0
        self.create_simulation_players()
        oneRound = Game(self.simulate_players)
        oneRound.deck.deck = self.simulated_undealed
        scoreDifference = oneRound.playOneRound(trick=copy.copy(self.trick),
                                                lead_player=copy.copy(self.lead_player),
                                                trickstartingTrick=copy.copy(startingTrick))

        # for i in range(0, len(self.dummy_players)):
        #    if i != self.dummy_p_idx:
        #        gain += (scoreDifference[self.dummy_p_idx] - scoreDifference[i])
        ## return simulate_players
        a = scoreDifference[self.dummy_p_idx]
        scoreDifference.pop(self.dummy_p_idx)
        b = max(scoreDifference)
        gain = a - b
        return gain

    # create simulated players and undealed hand
    def create_simulation_players(self):
        unknownHands = copy.copy(self.unknownHands)
        random.shuffle(unknownHands)
        for i in range(0, len(self.dummy_players)):
            self.simulate_players[i].name = copy.copy(self.dummy_players[i].name)
            self.simulate_players[i].hand = copy.copy(self.dummy_players[i].hand)
            self.simulate_players[i].score = copy.copy(self.dummy_players[i].score)
            self.simulate_players[i].zombie_count = copy.copy(self.dummy_players[i].zombie_count)
            # test = copy.deepcopy(self.dummy_players)
        # self.simulate_players = test
        # for i in range(0, len(self.dummy_players)):
        j = 0
        while j < len(self.dummy_players):
            if j != self.dummy_p_idx:
                # length = len(self.dummy_players[i].hand)
                # a = unknownHands[0:length]
                # self.simulate_players[i].hand = a
                # unknownHands = unknownHands[length:]

                length = len(self.dummy_players[j].hand)
                self.simulate_players[j].hand.clear()
                count = 0
                while self.playerHasNo[j][unknownHands[0][0]] and length > 0:
                    count = count + 1
                    if count > len(unknownHands):
                        unknownHands = copy.copy(self.unknownHands)
                        break
                    unknownHands.append(unknownHands.pop(0))
                if count > len(unknownHands):
                    j = 0
                    continue
                a = unknownHands[0:length]
                self.simulate_players[j].hand = a
                unknownHands = unknownHands[length:]
                random.shuffle(unknownHands)
            j = j + 1

        self.simulated_undealed = unknownHands

    def getAllUnknownHands(self):
        unknownHands = []
        # all unknown cards
        for i in range(0, len(self.dummy_players)):
            if i != self.dummy_p_idx:
                unknownHands.append(self.dummy_players[i].hand)
        unknownHands.append(self.undealed)
        unknownHands = sum(unknownHands, [])

        # shuffle unknown hands
        random.shuffle(unknownHands)
        self.unknownHands = unknownHands

    def allPossibleHands(self, trick):
        allPossibleHands = []
        if not trick:
            allPossibleHands = self.hand
        else:
            # find all the card that matches with the first one
            allPossibleHands = [hand for hand in self.hand if hand[0] == trick[0][0]]
        if not allPossibleHands:
            allPossibleHands = self.hand

        return allPossibleHands

    def playCard(self, trick, dummy_players, dummy_p_idx, undealed, lead_player, playerHasNo):

        print("-", self.name + "(" + str(self.score) + ")(Z" + str(self.zombie_count) + ")", "sees", trick)
        self.dummy_p_idx = dummy_p_idx
        self.dummy_players = dummy_players
        self.simulate_players = copy.deepcopy(dummy_players)
        self.undealed = undealed
        self.trick = trick
        self.lead_player = lead_player
        self.playerHasNo = playerHasNo
        self.getAllUnknownHands()

        self.rootNode = MctsNode(None)
        start_time = time.time()
        time_cost = 0
        while time_cost < self.time_limit:
            leafNode = self.selection()
            selectedNode = self.expansion(leafNode)
            value = self.simulation(selectedNode)
            self.backPropagation(selectedNode, value)
            time_cost = time.time() - start_time

        maxInvestigate = 0
        hand_to_play = self.rootNode.child[0].hand
        for i in range(len(self.rootNode.child)):
            if self.rootNode.child[i].investigate > maxInvestigate:
                maxInvestigate = self.rootNode.child[i].investigate
                hand_to_play = self.rootNode.child[i].hand
        if maxInvestigate >= 10:
            print('Reuse Occurs')
        self.hand.remove(hand_to_play)
        return hand_to_play

    def selection(self):
        currentNode = self.rootNode
        if len(currentNode.child) == 0:
            return currentNode
        else:
            maxUCB = currentNode.child[0].UCB
            next_idx = 0
            for i in range(len(currentNode.child)):
                if currentNode.child[i].UCB == None:
                    return currentNode.child[i]
                elif currentNode.child[i].UCB > maxUCB:
                    maxUCB = currentNode.child[i].UCB
                    next_idx = i
            return currentNode.child[next_idx]

    def expansion(self, node):
        if node == self.rootNode:
            allPossibleHands = self.allPossibleHands(self.trick)
            for hand in allPossibleHands:
                node.child.append(MctsNode(hand, node))
            return node.child[0]
        else:
            return node

    def simulation(self, node):
        value = self.oneRound(startingTrick=node.hand)
        return value

    def backPropagation(self, node, value):
        self.rootNode.investigate = self.rootNode.investigate + 1
        self.rootNode.value = self.rootNode.value + value
        self.rootNode.updateUCB(self.rootNode.investigate)
        while node.father != None:
            node.investigate = node.investigate + 1
            node.value = node.value + value
            node.updateUCB(self.rootNode.investigate)
            node = node.father

class MctsNode():
    def __init__(self, hand, father=None):
        self.hand = hand
        self.investigate = 0
        self.value = 0
        self.hand = hand
        self.father = father
        self.child = []
        self.UCB = None

    def updateUCB(self, totalInvestigate):
        if self.investigate == 0:
            self.UCB = None
        else:
            self.UCB = self.value / self.investigate + 2 * np.sqrt(np.log(totalInvestigate) / self.investigate)

class State():
    def __init__(self,dummy_players,trick, ai_id, leader_id):
        self.dummy_players = copy.deepcopy(dummy_players)
        self.trick = copy.deepcopy(trick)
        self.ai_id = ai_id
        self.leader_id = leader_id
        self.tag = 0
        #9 features may be  5
        self.zombie_army_score = dummy_players[ai_id].zombie_count
        #socre difference between AI and player with highest score
        self.score_difference = self.find_score_difference()
        self.score_difference2 = self.find_score_difference2()
        #total score for current game
        # self.score_difference2 = self.find_score_differenc2()
        #order of current trick our ai: 1 2 3
        self.order_of_AI = self.find_index_of_high_score()
        # index pf player who has highest score
        self.index_of_player_for_high_score = self.find_index_of_high_score()
        #total value of each suit
        self.z_total = self.find_Z_total()
        self.u_total = self.find_U_total()
        self.f_total = self.find_F_total()
        self.t_total = self.find_T_total()
        #How good a trick is
        self.trick_point = self.trick_worth()

    def trick_worth(self):
        if len(self.trick)==0:
            return 1
        elif len(self.trick)==1:
            if self.trick[0][0] == 'U':
                return 3
            elif self.trick[0][0] == 'F':
                return 2
            elif self.trick[0][0] == 'T':
                return 0
            else:
                return -1
        elif len(self.trick)==2:
            if self.trick[0][0] == "U" and self.trick[1][0]=="U":
                return 6
            elif self.trick[0][0] == "U" and self.trick[1][0]=="T":
                return 0
            elif self.trick[0][0] == "U" and self.trick[1][0]=="Z":
                return 2
            elif self.trick[0][0] == "U" and self.trick[1][0]=="F":
                return 5
            elif self.trick[0][0] == "T" and self.trick[1][0]=="T":
                return 0
            elif self.trick[0][0] == "T" and self.trick[1][0]=="U":
                return 0
            elif self.trick[0][0] == "T" and self.trick[1][0]=="Z":
                return -1
            elif self.trick[0][0] == "T" and self.trick[1][0]=="F":
                return 2
            elif self.trick[0][0] == "F" and self.trick[1][0]=="F":
                return 4
            elif self.trick[0][0] == "F" and self.trick[1][0]=="U":
                return 5
            elif self.trick[0][0] == "F" and self.trick[1][0]=="T":
                return 2
            elif self.trick[0][0] == "F" and self.trick[1][0]=="Z":
                return 1
            elif self.trick[0][0] == "Z" and self.trick[1][0]=="U":
                return 2
            elif self.trick[0][0] == "Z" and self.trick[1][0]=="Z":
                return -2
            elif self.trick[0][0] == "Z" and self.trick[1][0]=="F":
                return 1
            elif self.trick[0][0] == "Z" and self.trick[1][0]=="T":
                return -1

    def find_score_difference(self):
        score_difference = 0
        # for i in range(len(self.dummy_players)):
        #     max_score = max(max_score,self.dummy_players[i].score )
        list = []
        for i in range(len(self.dummy_players)):
            if i!=self.ai_id:
                list.append(self.dummy_players[i].score)

        score_difference = self.dummy_players[self.ai_id].score - list[0]
        return  score_difference

    def find_score_difference2(self):
        score_difference = 0
        # for i in range(len(self.dummy_players)):
        #     max_score = max(max_score,self.dummy_players[i].score )
        list = []
        for i in range(len(self.dummy_players)):
            if i != self.ai_id:
                list.append(self.dummy_players[i].score)
        score_difference = self.dummy_players[self.ai_id].score - list[1]
        return score_difference

    def find_order_in_this_trick(self):
        if self.ai_id==self.leader_id:
            return 1
        elif self.ai_id!=self.leader_id and self.ai_id>self.leader_id:
            return self.ai_id - self.leader_id + 1
        elif self.ai_id!=self.leader_id and self.ai_id<self.leader_id:
            return (self.ai_id - self.leader_id + len(self.dummy_players)) % len(self.dummy_players) + 1

    def find_index_of_high_score(self):
        max_score = -100
        index = -1
        for i in range(3):
            old_max_score = max_score
            max_score = max(max_score, self.dummy_players[i].score)
            if old_max_score!=max_score:
                index = i
        return index

    def find_T_total(self):
        hands = self.dummy_players[self.ai_id].hand
        total = 0
        if len(hands)!=0:
            for i in range(len(hands)):
                if hands[i][0] == 'T':
                    total += hands[i][1]
        return total

    def find_F_total(self):
        hands = self.dummy_players[self.ai_id].hand
        total = 0
        if len(hands)!=0:
            for i in range(len(hands)):
                if hands[i][0] == 'F':
                    total += hands[i][1]
        return total

    def find_Z_total(self):
        hands = self.dummy_players[self.ai_id].hand
        total = 0
        if len(hands)!=0:
            for i in range(len(hands)):
                if hands[i][0] == 'Z':
                    total += hands[i][1]
        return total
    def find_U_total(self):
        hands = self.dummy_players[self.ai_id].hand
        total = 0
        if len(hands)!=0:
            for i in range(len(hands)):
                if hands[i][0] == 'U':
                    total += hands[i][1]
        return total

class Action():
    def __init__(self, dummy_players, trick, hand_to_play,ai_idx):
        self.dummy_players = copy.deepcopy(dummy_players)
        self.trick = copy.deepcopy(trick)
        self.hand_to_play = copy.deepcopy(hand_to_play)
        self.ai_idx = ai_idx
        #features 6
        # if this hand could follow the trick 1, -1
        self.if_follow = self.can_follow()
        #win = 1, win with zombie = -0.5, loss = 1, unknow = 0.5
        self.win_tag = self.if_AI_win(ai_idx)
        # value change after AI play a hand
        # zombie changed value  after AI play a hand
        self.z_change = self.value_change('Z')
        # Unicorn changed value after AI play a hand
        self.u_change = self.value_change('U')
        # Fairy changed value  after AI play a hand
        self.f_change = self.value_change('F')
        # Troll changed value after AI play a hand
        self.t_change = self.value_change('T')

    def can_follow(self):
        if len(self.trick)==0:
            return 1
        else:
            suit = self.trick[0][0]
            card_idx = next((i for i, c in enumerate(self.dummy_players[self.ai_idx].hand) if c[0] == suit), None)
            if card_idx != None:
                return 1
            else:
                return -1
    #
    def if_AI_win(self, ai_inx):
        if len(self.trick) <2:
            return 0.5
        temp_trick = copy.deepcopy(self.trick)
        temp_trick.append(self.hand_to_play)
        winner_index,win_zombie= self.scoreTrick(temp_trick)
        if ai_inx!=winner_index:
            return -1
        if ai_inx==winner_index:
            if win_zombie:
                return -0.5
            return 1

    def suit_total(self, suit):
        hands = self.dummy_players[self.ai_id].hand
        total = 0
        if len(hands) != 0:
            for i in range(len(hands)):
                if hands[i][0] == suit:
                    total += hands[i][1]
        return total
    def value_change(self, suit):
        if self.hand_to_play[0] == suit:
            return self.hand_to_play[1]
        else:
            return 0

    def scoreTrick(self, trick):
        # Score the trick and add the score to the winning player
        # Get the suit led
        suit = trick[0][0]
        value = trick[0][1]
        winner = 0
        score = 0
        win_zombie = False
        # Determine who won (trick position not player!)
        for i in range(len(trick) - 1):
            if trick[i + 1][0] == suit and trick[i + 1][1] > value:
                winner = i + 1
                value = trick[i + 1][1]
        # Determine the score
        # Separate the suit and value tuples
        suits_list = list(zip(*trick))[0]
        if suits_list.count('T') == 0:
            # No Trolls, go ahead and score the unicorns
            score += suits_list.count('U') * 3
        score += suits_list.count('F') * 2
        n_zomb = suits_list.count('Z')
        if n_zomb > 0:
            win_zombie = True
        return winner ,win_zombie # Index of winning card


class Game():  # Main class
    def __init__(self, players):
        self.deck = Deck()
        self.players = players
        self.dummy_players = copy.deepcopy(players)
        self.played_cards = []  # List of already played cards
        # some constants
        self.HAND_SIZE = 18
        self.ZOMBIE_ARMY = 12
        self.ZOMBIE_ARMY_PENALTY = 20
        self.WIN_SCORE = 200
        self.episilon = 0.1

    def deal(self):
        self.deck.shuffle()
        self.played_cards = []
        for i in range(self.HAND_SIZE):
            for p in self.players:
                p.hand.append(self.deck.getCard())

    def scoreTrick(self, trick):
        # Score the trick and add the score to the winning player
        # Get the suit led
        suit = trick[0][0]
        value = trick[0][1]
        winner = 0
        score = 0
        # Determine who won (trick position not player!)
        for i in range(len(trick) - 1):
            if trick[i + 1][0] == suit and trick[i + 1][1] > value:
                winner = i + 1
                value = trick[i + 1][1]
        # Determine the score
        # Separate the suit and value tuples
        suits_list = list(zip(*trick))[0]
        if suits_list.count('T') == 0:
            # No Trolls, go ahead and score the unicorns
            score += suits_list.count('U') * 3
        score += suits_list.count('F') * 2
        n_zomb = suits_list.count('Z')
        score -= n_zomb
        return winner, score, n_zomb  # Index of winning card

    def play(self):
        lead_player = 0
        playerHasNo = []

        while True:  # Keep looping on hands until we have a winner
            self.deal()
            playerHasNo.clear()
            for k in range(len(self.players)):
                if type(self.players[k]).__name__ in ('RandomPlayer', 'GrabAndDuckPlayer', 'RolloutPlayer'):
                    continue
                ai_id = k
            for _ in range(len(self.players)):
                playerHasNo.append({'U': False, 'F': False, 'T': False, 'Z': False})
            while len(self.players[0].hand) > 0:
                trick = []
                # Form the trick, get a card from each player. Score the trick.
                for i in range(len(self.players)):
                    p_idx = (lead_player + i) % len(self.players)
                    # print(type(self.players[p_idx]).__name__)
                    if type(self.players[p_idx]).__name__ in ('RandomPlayer', 'GrabAndDuckPlayer'):
                        trick.append(self.players[p_idx].playCard(trick))
                    else:  # run AI assignment
                        # copy the current player list to local variable
                        # dummy_players = copy.deepcopy(self.players)
                        for i in range(0, len(self.dummy_players)):
                            self.dummy_players[i].name = copy.copy(self.players[i].name)
                            self.dummy_players[i].hand = copy.copy(self.players[i].hand)
                            self.dummy_players[i].score = copy.copy(self.players[i].score)
                            self.dummy_players[i].zombie_count = copy.copy(self.players[i].zombie_count)
                        dummy_p_idx = copy.deepcopy(p_idx)
                        undeald = copy.deepcopy(self.deck.deck)
                        dummy_lead_player = copy.deepcopy(lead_player)
                        if type(self.players[p_idx]).__name__ in ('RolloutPlayer', 'MCRLPlayer'):
                            played_trick = self.players[p_idx].playCard(trick, self.dummy_players, dummy_p_idx, undeald,dummy_lead_player, playerHasNo)
                        # elif type(self.players[p_idx]).__name__ in ('MCRLPlayer_No_Rollout'):
                        elif type(self.players[dummy_p_idx]).__name__ in ('MCRLPlayer_No_Rollout'):
                            played_trick = self.players[p_idx].QfunctionPlay(trick, 0, self.dummy_players, ai_id, p_idx, lead_player)
                        # print(self.players[p_idx].hand)
                        trick.append(played_trick)
                    if trick[-1][0] != trick[0][0]:
                        playerHasNo[p_idx][trick[0][0]] = True
                        print(self.players[p_idx].name, "run out of", trick[0][0])
                print(self.players[lead_player].name, "led:", trick)
                win_idx, score, n_zomb = self.scoreTrick(trick)

                # Convert winning trick index into new lead player index
                lead_player = (lead_player + win_idx) % len(self.players)
                print(self.players[lead_player].name, "won trick", score, "points")

                # Check for zombie army
                self.players[lead_player].zombie_count += n_zomb
                if self.players[lead_player].zombie_count >= self.ZOMBIE_ARMY:  # Uh-oh here comes the Zombie army!
                    self.players[lead_player].zombie_count = 0
                    print("***** ZOMBIE ARMY *****")
                    # Subtract 20 points from each opponent
                    for i in range(len(self.players) - 1):
                        self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY

                # Update score & check if won
                self.players[lead_player].score += score
                if self.players[lead_player].score >= self.WIN_SCORE:
                    print(self.players[lead_player].name, "won with", self.players[lead_player].score, "points!")
                    return self.players

                    # Keep track of the cards played
                self.played_cards.extend(trick)

            # Score the kitty (undealt cards)
            print(self.deck)
            win_idx, score, n_zomb = self.scoreTrick(self.deck.deck)
            print(self.players[lead_player].name, "gets", score, "points from the kitty")
            self.players[lead_player].score += score

            # Check for zombie army
            if self.players[lead_player].zombie_count + n_zomb >= self.ZOMBIE_ARMY:
                print("***** ZOMBIE ARMY *****")
                # Subtract 20 points from each opponent
                for i in range(len(self.players) - 1):
                    self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY

            # Check for winner
            if self.players[lead_player].score >= self.WIN_SCORE:
                print(self.players[lead_player].name, "won with", self.players[lead_player].score, "points!")
                return self.players

            print("\n* Deal a new hand! *\n")
            # reset the zombie count
            for p in self.players:
                p.zombie_count = 0

    def playOneRound(self, trick, lead_player, trickstartingTrick):
        intitial_score = []
        end_score = []
        difference = []
        trick = trick
        initialtart = 0
        ai_id = -1
        number_of_hand = 0
        for k in range(len(self.players)):
            if type(self.players[k]).__name__ in ('RandomPlayer', 'GrabAndDuckPlayer','RolloutPlayer'):
                continue
            ai_id = k
        dummy_players = copy.deepcopy(self.dummy_players)
        if not trick:
            initialtart = 1
        for player in self.players:
            intitial_score.append(player.score)
        while len(self.players[lead_player].hand) > 0:
            # Form the trick, get a card from each player. Score the trick.
            index_offset = 0
            for j in range(len(self.dummy_players)):
                dummy_players[j].name = copy.deepcopy(self.players[j].name)
                dummy_players[j].hand = copy.deepcopy(self.players[j].hand)
                dummy_players[j].score = copy.deepcopy(self.players[j].score)
                dummy_players[j].zombie_count = copy.copy(self.players[j].zombie_count)
            for i in range(len(self.players) - len(trick)):
                if i == 0 and (len(self.players) - len(
                        trick) != len(self.players) or initialtart == 1):  # in this case we are running the first play in round
                    index_offset = len(trick)
                    p_idx = (lead_player + i + index_offset) % len(self.players)
                    trick.append(self.players[p_idx].directPlay(trickstartingTrick))
                    initialtart = 0

                else:
                    p_idx = (lead_player + i + index_offset) % len(self.players)
                    # trick.append(self.players[p_idx].randomPlay(trick))
                    hand = self.players[p_idx].QfunctionPlay_rollout(trick, dummy_players[ai_id].epsilon, dummy_players, ai_id, p_idx, lead_player)
                    trick.append(hand)
            index_offset = 0
            # print(self.players[lead_player].name, "led:", trick)
            win_idx, score, n_zomb = self.scoreTrick(trick)

            # Convert winning trick index into new lead player index
            lead_player = (lead_player + win_idx) % len(self.players)
            # print(self.players[lead_player].name, "won trick", score, "points")
            # Check for zombie army
            self.players[lead_player].zombie_count += n_zomb
            if self.players[lead_player].zombie_count >= self.ZOMBIE_ARMY:  # Uh-oh here comes the Zombie army!
                self.players[lead_player].zombie_count = 0
                # print("***** ZOMBIE ARMY *****")
                # Subtract 20 points from each opponent
                for i in range(len(self.players) - 1):
                    self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY

            # Update score & check if won
            self.players[lead_player].score += score
            # if self.players[lead_player].score >= self.WIN_SCORE:
            #     print(self.players[lead_player].name, "won with", self.players[lead_player].score, "points!")
            #     return

            # Keep track of the cards played
            self.played_cards.extend(trick)
            trick = []
        # Score the kitty (undealt cards)
        # print(self.deck)
        win_idx, score, n_zomb = self.scoreTrick(self.deck.deck)
        # print(self.players[lead_player].name, "gets", score, "points from the kitty")
        self.players[lead_player].score += score
        # Check for zombie army
        if self.players[lead_player].zombie_count + n_zomb >= self.ZOMBIE_ARMY:
            # print("***** ZOMBIE ARMY *****")
            # Subtract 20 points from each opponent
            for i in range(len(self.players) - 1):
                self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY

        for player in self.players:
            end_score.append(player.score)

        zip_object = zip(end_score, intitial_score)
        for list1_i, list2_i in zip_object:
            difference.append(list1_i - list2_i)

        return difference

    def trainQfunction(self):
        coefficients = []

        pass

    def Qfunction(self,trick, lead_player, dummy_players, dummy_p_idx, undealed, playerHasNo, allPossibleHands):

        pass


class Game_Train():  # Main class
    def __init__(self, players):
        self.deck = Deck()
        self.players = players
        self.dummy_players = copy.deepcopy(players)
        self.played_cards = []  # List of already played cards
        # some constants
        self.HAND_SIZE = 18
        self.ZOMBIE_ARMY = 12
        self.ZOMBIE_ARMY_PENALTY = 20
        self.WIN_SCORE = 200
        self.episilon = 0.1
        self.features = None
    def deal(self):
        self.deck.shuffle()
        self.played_cards = []
        for i in range(self.HAND_SIZE):
            for p in self.players:
                p.hand.append(self.deck.getCard())

    def scoreTrick(self, trick):
        # Score the trick and add the score to the winning player
        # Get the suit led
        suit = trick[0][0]
        value = trick[0][1]
        winner = 0
        score = 0
        # Determine who won (trick position not player!)
        for i in range(len(trick) - 1):
            if trick[i + 1][0] == suit and trick[i + 1][1] > value:
                winner = i + 1
                value = trick[i + 1][1]
        # Determine the score
        # Separate the suit and value tuples
        suits_list = list(zip(*trick))[0]
        if suits_list.count('T') == 0:
            # No Trolls, go ahead and score the unicorns
            score += suits_list.count('U') * 3
        score += suits_list.count('F') * 2
        n_zomb = suits_list.count('Z')
        score -= n_zomb
        return winner, score, n_zomb  # Index of winning card
    def train(self):
        start = time.time()
        time_cost = 0
        count = 0
        print("start train")
        while time_cost<600:
            self.play()
            end = time.time()
            time_cost = end-start
            count +=1
            if count== 300:
                print("train 300 rollout...")
        print("count", count)
        print("slope",self.players[1].weight)
        print("bias", self.players[1].b)

    def play(self):
        lead_player = 0
        playerHasNo = []
        ai_id = -1
        number_of_hand = 0
        for k in range(len(self.players)):
            if type(self.players[k]).__name__ in ('RandomPlayer_Test', 'GrabAndDuckPlayer_Test'):
                continue
            ai_id = k
        # while True:  # Keep looping on hands until we have a winner
        self.deal()
        playerHasNo.clear()
        self.generate_random_socre(self.players)
        for _ in range(len(self.players)):
            playerHasNo.append({'U': False, 'F': False, 'T': False, 'Z': False})

        self.players[ai_id].win_score.clear()
        self.players[ai_id].y_label.clear()
        self.features = None
        while len(self.players[0].hand) > 0:
            trick = []
            # Form the trick, get a card from each player. Score the trick.
            for i in range(len(self.players)):
                p_idx = (lead_player + i) % len(self.players)
                # print(type(self.players[p_idx]).__name__)
                if type(self.players[p_idx]).__name__ in ('RandomPlayer_Test', 'GrabAndDuckPlayer_Test'):
                    trick.append(self.players[p_idx].playCard(trick))
                else:  # run AI assignment
                    ai_id = p_idx
                    # copy the current player list to local variable
                    # dummy_players = copy.deepcopy(self.players)
                    for i in range(0, len(self.dummy_players)):
                        self.dummy_players[i].name = copy.copy(self.players[i].name)
                        self.dummy_players[i].hand = copy.copy(self.players[i].hand)
                        self.dummy_players[i].score = copy.copy(self.players[i].score)
                        self.dummy_players[i].zombie_count = copy.copy(self.players[i].zombie_count)
                    dummy_p_idx = copy.deepcopy(p_idx)
                    undeald = copy.deepcopy(self.deck.deck)
                    dummy_lead_player = copy.deepcopy(lead_player)
                    # hand to play
                    played_trick, features_list = self.players[p_idx].Q_playCard_test(trick, self.dummy_players, dummy_p_idx, undeald,
                                                                dummy_lead_player, playerHasNo)
                    number_of_hand +=1
                    trick.append(played_trick)
                if trick[-1][0] != trick[0][0]:
                    playerHasNo[p_idx][trick[0][0]] = True
                    # print(self.players[p_idx].name, "run out of", trick[0][0])

            if self.features is None:
                self.features = np.array(features_list)
            else:
                self.features = np.append(self.features, features_list)
            # print(self.players[lead_player].name, "led:", trick)
            win_idx, score, n_zomb = self.scoreTrick(trick)
            # Convert winning trick index into new lead player index
            lead_player = (lead_player + win_idx) % len(self.players)
            # print(self.players[lead_player].name, "won trick", score, "points")
            # Check for zombie army
            self.players[lead_player].zombie_count += n_zomb
            if self.players[lead_player].zombie_count >= self.ZOMBIE_ARMY:  # Uh-oh here comes the Zombie army!
                self.players[lead_player].zombie_count = 0
                # print("***** ZOMBIE ARMY *****")
                # Subtract 20 points from each opponent
                for i in range(len(self.players) - 1):
                    self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY
            # update the score of ai in each trick
            if lead_player == ai_id:
                self.players[ai_id].win_score.append(score)
            else:
                self.players[ai_id].win_score.append(0)

            if self.players[lead_player].score >= self.WIN_SCORE:
                if len(self.players[ai_id].win_score)>5:
                    total_score = 0
                    temp_score = 0
                    for i in range(len(self.players[ai_id].win_score)):
                        total_score += self.players[ai_id].win_score[i]

                    for i in range(len(self.players[ai_id].win_score)):
                        if i == 0:
                            self.players[ai_id].y_label.append(total_score)
                            continue
                        temp_score += self.players[ai_id].win_score[i - 1]
                        self.players[ai_id].y_label.append(total_score - temp_score)

                    feature_single_list = np.array(self.features.tolist())
                    feature_matrix = feature_single_list.reshape(-1, 9)

                    # train our model use sklearn SGDRegression
                    # model = SGDRegressor()
                    # model.fit(feature_matrix, self.players[ai_id].y_label)
                    # update_w = model.coef_.tolist()
                    # count = 0
                    # for i in range(len(self.players[ai_id].weight)):
                    #     if model.coef_[i] == 0:
                    #         count += 1
                    # if count != 9:
                    #     self.players[ai_id].weight = update_w
                    #     self.players[ai_id].b = model.intercept_
                    ypredicted = np.zeros(len(feature_matrix))
                    for i in range(0, len(feature_matrix)):
                        for j in range(0, len(feature_matrix[0])):
                            ypredicted[i] += feature_matrix[i][j] * self.players[ai_id].weight[j]
                        ypredicted[i] += self.players[ai_id].b
                    stepsize = 1e-6
                    incept = 0
                    SGD = np.zeros(len(self.players[ai_id].weight))
                    for i in range(0, len(feature_matrix)):
                        for j in range(0, len(feature_matrix[0])):
                            SGD[j] += feature_matrix[i][j] * (self.players[ai_id].y_label[i] - ypredicted[i])
                            incept += self.players[ai_id].y_label[i] - ypredicted[i]
                    update_w = [self.players[ai_id].weight[i] + stepsize * SGD[i] for i in range(0, len(SGD))]
                    self.players[ai_id].weight = update_w
                    self.players[ai_id].b += stepsize * incept
                    # print("ypredicted", ypredicted)
                    # print("score", self.players[ai_id].win_score)
                    # print("label", self.players[ai_id].y_label)
                    # print("slope", self.players[ai_id].weight)
                    # print("bias", self.players[ai_id].b)
                # print(self.players[lead_player].name, "won with", self.players[lead_player].score, "points!")
                return self.players
                # Keep track of the cards played
            self.played_cards.extend(trick)

        # Score the kitty (undealt cards)
        # print(self.deck)
        win_idx, score, n_zomb = self.scoreTrick(self.deck.deck)
        # print(self.players[lead_player].name, "gets", score, "points from the kitty")
        self.players[lead_player].score += score
        # Check for zombie army
        if self.players[lead_player].zombie_count + n_zomb >= self.ZOMBIE_ARMY:
            # print("***** ZOMBIE ARMY *****")
            # Subtract 20 points from each opponent
            for i in range(len(self.players) - 1):
                self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY
        if lead_player==ai_id:
            self.players[ai_id].win_score[-1]+=score
        # update y_label
        total_score = 0
        temp_score = 0
        for i in range(len(self.players[ai_id].win_score)):
            total_score += self.players[ai_id].win_score[i]

        for i in range(len(self.players[ai_id].win_score)):
            if i == 0:
                self.players[ai_id].y_label.append(total_score)
                continue
            temp_score += self.players[ai_id].win_score[i-1]
            self.players[ai_id].y_label.append(total_score - temp_score)

        feature_single_list = np.array(self.features.tolist())
        feature_matrix =  feature_single_list.reshape(-1,9)

        #train our model use sklearn SGDRegression
        # model = SGDRegressor()
        # model.fit(feature_matrix, self.players[ai_id].y_label)
        # update_w = model.coef_.tolist()
        # count = 0
        # for i in range(len(self.players[ai_id].weight)):
        #     if model.coef_[i] ==0:
        #         count +=1
        # if count!=9:
        #     self.players[ai_id].weight = update_w
        #     self.players[ai_id].b = model.intercept_
        #     # print("score", self.players[ai_id].win_score)
        #     # print("label", self.players[ai_id].y_label)
        #     # print("slope", self.players[ai_id].weight)
        #     # print("bias", model.intercept_)
        ypredicted = np.zeros(len(feature_matrix))
        for i in range(0, len(feature_matrix)):
            for j in range(0, len(feature_matrix[0])):
                ypredicted[i] += feature_matrix[i][j] * self.players[ai_id].weight[j]
            ypredicted[i] += self.players[ai_id].b
        stepsize = 1e-6
        incept = 0
        SGD = np.zeros(len(self.players[ai_id].weight))
        for i in range(0, len(feature_matrix)):
            for j in range(0, len(feature_matrix[0])):
                SGD[j] += feature_matrix[i][j] * (self.players[ai_id].y_label[i] - ypredicted[i])
                incept += self.players[ai_id].y_label[i] - ypredicted[i]
        update_w = [self.players[ai_id].weight[i] + stepsize * SGD[i] for i in range(0, len(SGD))]
        self.players[ai_id].weight = update_w
        self.players[ai_id].b += stepsize * incept
        # print("ypredicted", ypredicted)
        # print("score", self.players[ai_id].win_score)
        # print("label", self.players[ai_id].y_label)
        # print("slope", self.players[ai_id].weight)
        # print("bias", self.players[ai_id].b)
        # Check for winne
        if self.players[lead_player].score >= self.WIN_SCORE:
            # print(self.players[lead_player].name, "won with", self.players[lead_player].score, "points!")
            return self.players

        # print("\n* Deal a new hand! *\n")
        # reset the zombie count
        for p in self.players:
            p.zombie_count = 0

     # generate random score for each player in one trick

    def generate_random_socre(self, dummy_players):

        one_player = random.randint(-10, 199)
        two_player = random.randint(-10, 199)
        three_player = random.randint(-10,199)
        # print(one_player, two_player,three_player)
        dummy_players[0].score = one_player
        dummy_players[1].score = two_player
        dummy_players[2].score = three_player
    def playOneRound(self, trick, lead_player, trickstartingTrick):
        intitial_score = []
        end_score = []
        difference = []
        trick = trick
        initialtart = 0
        if not trick:
            initialtart = 1
        for player in self.players:
            intitial_score.append(player.score)
        while len(self.players[lead_player].hand) > 0:
            # Form the trick, get a card from each player. Score the trick.
            index_offset = 0
            for i in range(len(self.players) - len(trick)):
                if i == 0 and (len(self.players) - len(
                        trick) != len(
                    self.players) or initialtart == 1):  # in this case we are running the first play in round
                    index_offset = len(trick)
                    p_idx = (lead_player + i + index_offset) % len(self.players)
                    trick.append(self.players[p_idx].directPlay(trickstartingTrick))
                    initialtart = 0
                else:
                    p_idx = (lead_player + i + index_offset) % len(self.players)
                    trick.append(self.players[p_idx].randomPlay(trick))
            index_offset = 0
            # print(self.players[lead_player].name, "led:", trick)
            win_idx, score, n_zomb = self.scoreTrick(trick)
            # Convert winning trick index into new lead player index
            lead_player = (lead_player + win_idx) % len(self.players)
            # print(self.players[lead_player].name, "won trick", score, "points")
            # Check for zombie army
            self.players[lead_player].zombie_count += n_zomb
            if self.players[lead_player].zombie_count >= self.ZOMBIE_ARMY:  # Uh-oh here comes the Zombie army!
                self.players[lead_player].zombie_count = 0
                # print("***** ZOMBIE ARMY *****")
                # Subtract 20 points from each opponent
                for i in range(len(self.players) - 1):
                    self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY

            # Update score & check if won
            self.players[lead_player].score += score
            # if self.players[lead_player].score >= self.WIN_SCORE:
            #     print(self.players[lead_player].name, "won with", self.players[lead_player].score, "points!")
            #     return

            # Keep track of the cards played
            self.played_cards.extend(trick)
            trick = []

        # Score the kitty (undealt cards)
        # print(self.deck)
        win_idx, score, n_zomb = self.scoreTrick(self.deck.deck)
        # print(self.players[lead_player].name, "gets", score, "points from the kitty")
        self.players[lead_player].score += score
        # Check for zombie army
        if self.players[lead_player].zombie_count + n_zomb >= self.ZOMBIE_ARMY:
            # print("***** ZOMBIE ARMY *****")
            # Subtract 20 points from each opponent
            for i in range(len(self.players) - 1):
                self.players[(lead_player + 1 + i) % len(self.players)].score -= self.ZOMBIE_ARMY_PENALTY

        for player in self.players:
            end_score.append(player.score)

        zip_object = zip(end_score, intitial_score)
        for list1_i, list2_i in zip_object:
            difference.append(list1_i - list2_i)

        return difference

    def trainQfunction(self):
        coefficients = []

        pass

    def Qfunction(self, trick, lead_player, dummy_players, dummy_p_idx, undealed, playerHasNo, allPossibleHands):

        pass





