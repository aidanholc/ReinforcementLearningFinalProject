from pokerkit import *
import numpy as np

class PokerEnvironment:
    def __init__(self, config):
        self.config = config
        self.n_samples = config['monte_carlo_sample_size']

        # Start hand
        self.game_state = NoLimitTexasHoldem.create_state(
            automations=(
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING
            ),
            ante_trimming_status=False,
            raw_antes=0,
            raw_blinds_or_straddles=(1,2),
            min_bet=2,
            raw_starting_stacks=(self.config['player_0_stack'], self.config['player_1_stack']),
            player_count=2,
            mode = Mode.CASH_GAME
        )

        # Initialize Variables
        self.raises = [0, 0]
        self.calls = [0, 0]
        self.played_hands = [0, 0]
        self.total_hands = 0

        # self.is_complete = False
        # self.has_played_current_hand = [False, False]
        # self.hand_state = 0
        # self.total_hands += 1
        # # Deal Preflop Cards
        # for i in range(4):
        #     self.game_state.deal_hole()
        # self.pot_size = self.game_state.total_pot_amount
        # self.player_0_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[0], self.game_state.board_cards, self.n_samples)
        # self.player_1_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[1], self.game_state.board_cards, self.n_samples)


    def reset(self):
        stacks = self.game_state.stacks
        if stacks[0] == 0:
            print('PLAYER 0 RAN OUT OF MONEY')
            stacks[0] = self.config['player_0_stack']
            stacks[1] = self.config['player_1_stack']
        elif stacks[1] == 0:
            print('PLAYER 1 RAN OUT OF MONEY')
            stacks[1] = self.config['player_1_stack']
            stacks[0] = self.config['player_0_stack']
        self.game_state = NoLimitTexasHoldem.create_state(
            automations=(
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING
            ),
            ante_trimming_status=False,
            raw_antes=0,
            raw_blinds_or_straddles=(1,2),
            min_bet=2,
            raw_starting_stacks=(stacks[0], stacks[1]),
            player_count=2,
            mode = Mode.CASH_GAME
        )

            # self.is_complete = False
            # self.has_played_current_hand = [False, False]
            # self.hand_state = 0
            # self.total_hands += 1
            # # Deal Preflop Cards
            # for i in range(4):
            #     self.game_state.deal_hole()
            # self.pot_size = self.game_state.total_pot_amount
            # self.player_0_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[0], self.game_state.board_cards, self.n_samples)
            # self.player_1_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[1], self.game_state.board_cards, self.n_samples)

        
    def get_state(self, player_index):
        if player_index < 0 or player_index > 1 :
            raise Exception('Player Index can only be 0 or 1')
        hand_state = self.hand_state # are we preflop, flop, turn, or river
        bets = self.game_state.bets
        pot_size = self.pot_size
        if player_index == 0:
            hand_strength = self.player_0_hand_strength
            money_to_call = max(bets[1] - bets[0], 0)
            stack = self.game_state.stacks[0]
            af = self._calculate_af(self.raises[1], self.calls[1]) # calculate player 1's af
            lh = self._calculate_lh(self.played_hands[1])
        else:
            hand_strength = self.player_1_hand_strength
            money_to_call = max(bets[0] - bets[1], 0)
            stack = self.game_state.stacks[1]
            af = self._calculate_af(self.raises[0], self.calls[0]) # calculate player 0's af
            lh = self._calculate_lh(self.played_hands[0])
        
        return {'stack': stack,
                'pot_size': pot_size,
                'money_to_call': money_to_call,
                'hand_strength': hand_strength,
                'af': af,
                'lh': lh,
                'hand_state': hand_state}
    
    def get_whos_turn(self):
        return self.game_state.turn_index
    

    def _calculate_af(self, raises, calls):
        if calls + raises == 0:
            af = 0.5
        else:
            af = raises / (calls + raises)
        return af
    
    def _calculate_lh(self, played_hands):
        if self.total_hands < 10:
            lh = 1#0.75
        else:
            lh = played_hands / self.total_hands
        return lh
    

    def _calculate_hand_strength(self, hand, board, n_samples):
        hand = frozenset([frozenset(hand)])
        #print(hand)
        board = frozenset([x[0] for x in board])
        #print(board)
        strength = calculate_hand_strength(
            player_count= 2,
            hole_range = hand,
            board_cards = board,
            hole_dealing_count=2,
            board_dealing_count=5,
            deck=Deck.STANDARD,
            hand_types=(StandardHighHand,),
            sample_count=n_samples,
            #executor=executor
        )
        return strength


    def _calculate_hand_potential(self, hand, board, n_samples):
        return 0.5
    

    def _calculate_opponent_hand_potential(self, your_hand, board, outer_sim_samples, inner_sim_samples):
        potentials = list()
        for i in range(outer_sim_samples):
            potentials.append(self._calculate_hand_potential(your_hand, board, inner_sim_samples))
        return np.mean(potentials)

    def deal_hands(self):
        """
        SHOULD ONLY CALL DIRECTLY AFTER RESET OR INITIALIZATION.
        """
        self.is_complete = False
        self.has_played_current_hand = [False, False]
        self.hand_state = 0
        self.total_hands += 1
        # Deal Preflop Cards
        for i in range(4):
            self.game_state.deal_hole()
        self.pot_size = self.game_state.total_pot_amount
        self.player_0_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[0], self.game_state.board_cards, self.n_samples)
        self.player_1_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[1], self.game_state.board_cards, self.n_samples)

        # player_0_state = self.get_state(0)
        # player_1_state = self.get_state(1)

        # player_0_reward = self.reward(player_0_state, 0)
        # player_1_reward = self.reward(player_1_state, 1)

        # return [{'done': self.is_complete, 'state': player_0_state, 'reward': player_0_reward}, 
        #         {'done': self.is_complete, 'state': player_1_state, 'reward': player_1_reward}]

    def step(self, action, player_index):  

        started_complete = self.is_complete   
        if started_complete:
            print('RAN INTO STEP WHEN THE GAME WAS ALREADY COMPLETE')
            new_state = self.get_state(player_index)
            reward = self.reward(state, None, player_index)
            return {'done': self.is_complete, 'state': new_state,'reward': reward}

        # Ensuring action is valid
        if int(action['fold']) + int(action['check_or_call']) + int(action['raise']) != 1:
            raise Exception(f'Too many actions selected: {action}')

        # Handle Folds
        if not action['fold']:
            self.has_played_current_hand[player_index] = True
        else:
            self.game_state.fold()
            self.is_complete = True

        # Handle Checks or Calls 
        if action['check_or_call']:
            output = self.game_state.check_or_call()
            self.calls[player_index] += 1
            self.pot_size += output.amount

        # Handle Raises
        if action['raise']:
            output = self.game_state.complete_bet_or_raise_to(action['bet_size'])
            self.raises[player_index] += 1
            self.pot_size += output.amount

        
        # Check if game is over
        is_player_0_in_hand = self.game_state.statuses[0]
        is_player_1_in_hand = self.game_state.statuses[1]
        if not (is_player_0_in_hand and is_player_1_in_hand):
            self.is_complete = True

        # Handle if players are all in
        if self.game_state.all_in_status:
            self.pot_size = self.game_state.total_pot_amount
            if self.hand_state != 3: # Deal more cards if there's more cards to deal
                self.game_state.select_runout_count(1)
                self.game_state.select_runout_count(1)
                while self.game_state.can_burn_card():
                    self.hand_state += 1
                    self.game_state.burn_card()
                    self.game_state.deal_board()
                # Update hand strengths based on dealt cards
                self.player_0_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[0], self.game_state.board_cards, self.n_samples)
                self.player_1_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[1], self.game_state.board_cards, self.n_samples)
            self.is_complete = True

        if self.is_complete:
            if self.has_played_current_hand[0]:
                self.played_hands[0] += 1
            if self.has_played_current_hand[1]:
                self.played_hands[1] += 1
            

        # Increment hand state
        if self.game_state.can_burn_card():
            self.hand_state += 1
            self.game_state.burn_card()
            self.game_state.deal_board()
            # Update hand strengths based on new cards 
            self.player_0_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[0], self.game_state.board_cards, self.n_samples)
            self.player_1_hand_strength = self._calculate_hand_strength(self.game_state.hole_cards[1], self.game_state.board_cards, self.n_samples)

        new_state = self.get_state(player_index)
        reward = self.reward(new_state, None, player_index)
        
        return {'done': self.is_complete, 'state': new_state,'reward': reward}


    def reward(self, state, action, player_index):
        # Handle reward if user won/lost
        epsilon = 0.001
        if player_index == 0:
            if self.is_complete:
                did_win = self.game_state.statuses[player_index]
                if did_win:
                    return state['pot_size']
                else:
                    return -1 * state['pot_size']
            # Handle reward if mid hand
            return ((state['hand_strength'] ** (1/(state['lh'] + epsilon))) - (1 - (state['hand_strength'] ** (1/(state['lh'] + epsilon))))) * state['pot_size']
        if player_index == 1:
            if action is not None and action == 'fold':
                return -1 * ((state['hand_strength'] ** (1/(state['lh'] + epsilon))) - (1 - (state['hand_strength'] ** (1/(state['lh'] + epsilon))))) * state['pot_size']
            if self.is_complete:
                did_win = self.game_state.statuses[player_index]
                if did_win:
                    return state['pot_size']
                else:
                    return -1 * state['pot_size']
            # Handle reward if mid hand
            epsilon = 0.001
            return ((state['hand_strength'] ** (1/(state['lh'] + epsilon))) - (1 - (state['hand_strength'] ** (1/(state['lh'] + epsilon))))) * state['pot_size']