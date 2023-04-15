##################################
## CS4386 Semester B, 2022-2023
## Assignment 1
## Name: WANG Yian
## Student ID: 56641105 
##################################


import copy 
from math import inf as infinity
import random

import numpy as np


class ConstantWya:
    P1_MAX = "X"
    P2_MIN = "O"
    FINAL_SCORE_C = 0.75
    RULE_DEPTH = 5 #NOTE: need to be a odd number
    NEG_INF = -np.inf
    POS_INF = np.inf
    MIN_BRANCH_NUM = 2
    UCB_C = 5
    MCT_N = 12500

    PLUS_L1 = 2
    PLUS_L2 = 9
    PLUS_L3 = 12
    MINUS_L1 = -3
    MINUS_L2 = -12


class StateWya(object):
    def __init__(self, state: np.ndarray, player, h_score):
        self.state = state
        self.player = player # NOTE: the one who play the next move ("X" or "O")
        self.h_score = h_score

    def is_terminate(self):
        return len(np.where(self.state == None)[0]) == 0
    
    def empty_block(self):
        return len(np.where(self.state == None)[0])
    
    def __str__(self) -> str:
        return self.player + " " +str(self.h_score)


class PointWya(object):
    def __init__(self, coordinate: tuple, player):
        """
        coordinate: (x, y)
        player: "X", "O"
        """
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.coordinate = coordinate
        self.player = player

    def __str__(self) -> str:
        return str((self.coordinate, self.player))


class MCTNodeWya(object):
    def __init__(self, parent=None, state:StateWya=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None
def ucb(n, C=ConstantWya.UCB_C):
    return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)


class AIPlayer(object):
    def __init__(self, name, symbole, isAI=False):
        self.name = name
        self.symbole = symbole
        self.isAI = isAI
        self.score=0

    def stat(self):
        return self.name + " won " + str(self.won_games) + " games, " + str(self.draw_games) + " draw."

    def __str__(self):
        return self.name
    def get_isAI(self):
        return self.isAI
    def get_symbole(self):
        return self.symbole
    def get_score(self):
        return self.score
    def add_score(self,score):
        self.score+=score
    
    def available_cells(self,state,player):
        """
        params:
            state: 6*6 2d list to represent the board, eg. for a row: ['X' None 'O' None None None] ("X": p1; "O": p2)

        returns: the current available cells of the board regardless of players
        """
        cells = []

        for x, row in enumerate(state):
            for y, cell in enumerate(row):
                if (cell is None):
                    cells.append([x, y])
        return cells
    

    def available_cells_color(self, state, player):
        # NOTE: p1-black-X-even; p2-white-O-odd
        # print(colors.RED, "state", state, colors.ENDC)
        player = 0 if player == "X" else 1
        cells = []
        for x, row in enumerate(state):
            for y, cell in enumerate(row):
                if (cell == None) and (x+y+player)%2 == 0:
                    cells.append([x, y])
        return cells
    
    # original get_move function
    def get_move_ori(self, state, player) -> list:
        """
        params: 
            player: player1: "X"; player2: "O"
        returns: next move in the format of [row, col]
        """
        valid_actions = self.available_cells_color(state, player)
        random_move=random.choice(valid_actions)
        print("\n\n\n\n")
        return random_move   
    


    def get_same_block_around(self, pos, is_row):
        """
        is_row: get blocks with same color on the same row
        """
        ret = []
        if is_row:
            if (pos[1]-2) >= 0: ret.append((pos[0], pos[1] - 2))
            if (pos[1]+2) <= 5: ret.append((pos[0], pos[1] + 2))
        else:
            if (pos[0]-2) >= 0: ret.append((pos[0] - 2, pos[1]))
            if (pos[0]+2) <= 5: ret.append((pos[0] + 2, pos[1]))
        return ret
    
    def get_different_block_around(self, pos, is_row):
        ret = []
        if is_row:
            if (pos[1]-1) >= 0: ret.append((pos[0], pos[1] - 1))
            if (pos[1]+1) <= 5: ret.append((pos[0], pos[1] + 1))
        else:
            if (pos[0]-1) >= 0: ret.append((pos[0] - 1, pos[1]))
            if (pos[0]+1) <= 5: ret.append((pos[0] + 1, pos[1]))
        return ret
    
    def add_point_vals(self, point_vals, point, value):
        if point_vals.get(point) == None: point_vals[point] = [0, 0] # NOTE: (neg, pos)
        point_vals[point][int(value > 0)] += value
        if point_vals[point][int(value > 0)] != value:
            point_vals[point][int(value > 0)] = ConstantWya.FINAL_SCORE_C * (point_vals[point][int(value > 0)])

    def find_point(self, points: list, coordinate):
        """
        find a point in a point list by coordinate

        returns:
            if found, return the point; if not, return None
        """

        for p in points:
            if p.coordinate == coordinate: return p
        return None
    
    def on_board(self, nums: list):
        """
        check all nums in list are available
        """
        for n in nums:
            if n<0 or n>5: return False
        return True

    def evaluate_points(self, point_vals, this_row, this_col, player, cur_point: PointWya):
        """
        params:
            player: the player is playing (in the perspective of THIS player)
            cur_point: the point is being analyzed
        
        No matter which current player it is, the score is always the greater the better for this player
        """

        self_on_this_row = []
        self_on_this_col = []
        oppo_on_this_row = []
        oppo_on_this_col = []
        self_row_cnt = self_col_cnt = oppo_row_cnt = oppo_col_cnt = 0

        for t in this_row:
            if t.player == player:
                self_row_cnt += 1
                self_on_this_row.append(t)
            else:
                oppo_row_cnt += 1
                oppo_on_this_row.append(t)
        for t in this_col:
            if t.player == player:
                self_col_cnt += 1
                self_on_this_col.append(t)
            else:
                oppo_col_cnt += 1
                oppo_on_this_col.append(t)

        if cur_point.player == player:
            # -6 rule
            if self_row_cnt == 2 and oppo_row_cnt != 3:
                p = (self_on_this_row[0].x, 6+(self_on_this_row[0].y%2)*3 - self_on_this_row[0].y - self_on_this_row[1].y) # NOTE: the only empty same color block on this row
                self.add_point_vals(point_vals, p, ConstantWya.MINUS_L2)
            if self_col_cnt == 2 and oppo_col_cnt != 3:
                p = (6+(self_on_this_col[0].x%2)*3 - self_on_this_col[0].x - self_on_this_col[1].x, self_on_this_col[0].y) # NOTE: the only empty same color block on this row
                self.add_point_vals(point_vals, p, ConstantWya.MINUS_L2)

            # +6 rule
            if self_row_cnt == 2 and oppo_row_cnt == 3:
                p = (self_on_this_row[0].x, 6+(self_on_this_row[0].y%2)*3 - self_on_this_row[0].y - self_on_this_row[1].y)
                self.add_point_vals(point_vals, p, ConstantWya.PLUS_L3)
            if self_col_cnt == 2 and oppo_col_cnt == 3:
                p = (6+(self_on_this_col[0].x%2)*3 - self_on_this_col[0].x - self_on_this_col[1].x, self_on_this_col[0].y) # NOTE: the only empty same color block on this row
                self.add_point_vals(point_vals, p, ConstantWya.PLUS_L3)

            # +3 rule
            if (self_row_cnt >= 1 and self_row_cnt <= 3) and oppo_row_cnt >= 1:
                for s in self_on_this_row:
                    if (self.find_point(oppo_on_this_row, (s.x, s.y+1)) 
                        and not self.find_point(oppo_on_this_row, (s.x, s.y-1)) 
                        and not self.find_point(self_on_this_row, (s.x, s.y+2))
                        and self.on_board([s.y+1, s.y+2])):
                        self.add_point_vals(point_vals, (s.x, s.y+2), ConstantWya.PLUS_L2)
                    elif (not self.find_point(oppo_on_this_row, (s.x, s.y+1)) 
                        and self.find_point(oppo_on_this_row, (s.x, s.y-1)) 
                        and not self.find_point(self_on_this_row, (s.x, s.y-2))
                        and self.on_board([s.y-1, s.y-2])):
                        self.add_point_vals(point_vals, (s.x, s.y-2), ConstantWya.PLUS_L2)
            if (self_col_cnt >= 1 and self_col_cnt <= 3) and oppo_col_cnt >= 1:
                for s in self_on_this_col:
                    if (self.find_point(oppo_on_this_col, (s.x+1, s.y)) 
                        and not self.find_point(oppo_on_this_col, (s.x-1, s.y)) 
                        and not self.find_point(self_on_this_col, (s.x+2, s.y))
                        and self.on_board([s.x+1, s.x+2])):
                        self.add_point_vals(point_vals, (s.x+2, s.y), ConstantWya.PLUS_L2)
                    elif (not self.find_point(oppo_on_this_col, (s.x+1, s.y)) 
                        and self.find_point(oppo_on_this_col, (s.x-1, s.y)) 
                        and not self.find_point(self_on_this_col, (s.x-2, s.y))
                        and self.on_board([s.x-1, s.x-2])):
                        self.add_point_vals(point_vals, (s.x-2, s.y), ConstantWya.PLUS_L2)

            # -3 rule
            if self_row_cnt == 1: # NOTE: only itself on this row
                for p in self.get_same_block_around(cur_point.coordinate, True):
                    if point_vals.get(p) == None or point_vals.get(p)[1] <= 4:
                        self.add_point_vals(point_vals, p, -3)
            if self_col_cnt == 1: # NOTE: only itself on this col
                for p in self.get_same_block_around(cur_point.coordinate, False):
                    if point_vals.get(p) == None or point_vals.get(p)[1] <= 4:
                        self.add_point_vals(point_vals, p, -3)

        else:
            # +2? # NOTE: defending logic
            if oppo_row_cnt == 1:
                if cur_point.y+3 <= 5: self.add_point_vals(point_vals, (cur_point.x, cur_point.y+3), ConstantWya.PLUS_L1/2)
                if cur_point.y-3 >= 0: self.add_point_vals(point_vals, (cur_point.x, cur_point.y-3), ConstantWya.PLUS_L1/2)
            if oppo_col_cnt == 1:
                if cur_point.x+3 <= 5: self.add_point_vals(point_vals, (cur_point.x+3, cur_point.y), ConstantWya.PLUS_L1/2)
                if cur_point.x-3 <= 0: self.add_point_vals(point_vals, (cur_point.x-3, cur_point.y), ConstantWya.PLUS_L1/2)

            # +3 # NOTE: oppo empty oppo type detection
            if oppo_row_cnt == 2 and self_row_cnt <= 1:
                if (abs(oppo_on_this_row[0].y - oppo_on_this_row[1].y) == 2
                    and (self_row_cnt == 0
                        or not (self.find_point(self_on_this_row, (oppo_on_this_row[0].x, oppo_on_this_row[0].y-1))
                            or self.find_point(self_on_this_row, (oppo_on_this_row[0].x, oppo_on_this_row[0].y+1))
                            or self.find_point(self_on_this_row, (oppo_on_this_row[1].x, oppo_on_this_row[1].y-1))
                            or self.find_point(self_on_this_row, (oppo_on_this_row[1].x, oppo_on_this_row[1].y+1))
                        ))
                    ):
                    self.add_point_vals(point_vals, (oppo_on_this_row[0].x, int(0.5*(oppo_on_this_row[0].y + oppo_on_this_row[1].y))), ConstantWya.PLUS_L2)
            if oppo_col_cnt == 2 and self_col_cnt <= 1:
                if (abs(oppo_on_this_col[0].x - oppo_on_this_col[1].x) == 2
                    and (self_col_cnt == 0
                        or not (self.find_point(self_on_this_col, (oppo_on_this_col[0].x-1, oppo_on_this_col[0].y))
                            or self.find_point(self_on_this_col, (oppo_on_this_col[0].x+1, oppo_on_this_col[0].y))
                            or self.find_point(self_on_this_col, (oppo_on_this_col[1].x-1, oppo_on_this_col[1].y))
                            or self.find_point(self_on_this_col, (oppo_on_this_col[1].x+1, oppo_on_this_col[1].y))
                        ))
                    ):
                    self.add_point_vals(point_vals, (int(0.5*(oppo_on_this_col[0].x + oppo_on_this_col[1].x)), oppo_on_this_row[0].y), ConstantWya.PLUS_L2)

            # +2？# NOTE: attacking logic
            if oppo_row_cnt in [1, 2]:
                if ((1 in [p.y for p in oppo_on_this_row] 
                    and not 3 in [p.y for p in oppo_on_this_row])
                    or (4 in [p.y for p in oppo_on_this_row]
                    and not 2 in [p.y for p in oppo_on_this_row])):
                    self.add_point_vals(point_vals, (oppo_on_this_row[0].x, 0), ConstantWya.PLUS_L2)
                    self.add_point_vals(point_vals, (oppo_on_this_row[0].x, 5), ConstantWya.PLUS_L2)
            if oppo_col_cnt in [1, 2]:
                if ((1 in [p.x for p in oppo_on_this_col] 
                    and not 3 in [p.x for p in oppo_on_this_col])
                    or (4 in [p.x for p in oppo_on_this_col]
                    and not 2 in [p.x for p in oppo_on_this_col])):
                    self.add_point_vals(point_vals, (0, oppo_on_this_col[0].y), ConstantWya.PLUS_L2)
                    self.add_point_vals(point_vals, (5, oppo_on_this_col[0].y), ConstantWya.PLUS_L2)

            # -3
            if oppo_row_cnt == 1: # NOTE: only itself on this row
                for p in self.get_different_block_around(cur_point.coordinate, True):
                    if point_vals.get(p) == None or point_vals.get(p)[1] <= 4:
                        self.add_point_vals(point_vals, p, ConstantWya.MINUS_L1)
            if oppo_col_cnt == 1: # NOTE: only itself on this col
                for p in self.get_different_block_around(cur_point.coordinate, False):
                    if point_vals.get(p) == None or point_vals.get(p)[1] <= 4:
                        self.add_point_vals(point_vals, p, ConstantWya.MINUS_L1)

        return point_vals

    def get_branches(self, player, p1_pos, p2_pos):
        row_set = {}
        col_set = {}
        for p in p1_pos:
            if row_set.get(p[0]) == None: row_set[p[0]] = []
            row_set[p[0]].append(PointWya(p, ConstantWya.P1_MAX))
            if col_set.get(p[1]) == None: col_set[p[1]] = []
            col_set[p[1]].append(PointWya(p, ConstantWya.P1_MAX))

        for p in p2_pos:
            if row_set.get(p[0]) == None: row_set[p[0]] = []
            row_set[p[0]].append(PointWya(p, ConstantWya.P2_MIN))
            if col_set.get(p[1]) == None: col_set[p[1]] = []
            col_set[p[1]].append(PointWya(p, ConstantWya.P2_MIN))

        point_vals = {}
        # +2 rule for corner
        if player == ConstantWya.P1_MAX:
            self.add_point_vals(point_vals, (0, 0), ConstantWya.PLUS_L1)
            self.add_point_vals(point_vals, (5, 5), ConstantWya.PLUS_L1)
        else:
            self.add_point_vals(point_vals, (0, 5), ConstantWya.PLUS_L1)
            self.add_point_vals(point_vals, (5, 0), ConstantWya.PLUS_L1)

        for p in p1_pos:
            point_vals = self.evaluate_points(point_vals, row_set[p[0]], col_set[p[1]], player, PointWya(p, ConstantWya.P1_MAX))
        for p in p2_pos:
            point_vals = self.evaluate_points(point_vals, row_set[p[0]], col_set[p[1]], player, PointWya(p, ConstantWya.P2_MIN))

        return point_vals

    # NOTE: p1-black-X-even; p2-white-O-odd
    def get_child_actions(self, node: StateWya):
        p1_pos = list(zip(*np.where(node.state == ConstantWya.P1_MAX)))
        p2_pos = list(zip(*np.where(node.state == ConstantWya.P2_MIN)))

        point_vals = self.get_branches(node.player, p1_pos, p2_pos)

        # NOTE: point_vals is a dictionary action(coordinate): score
        # NOTE: sort by vals

        empty_blocks = set(t for t in zip(*np.where(node.state == None)) if sum(t)%2 == (node.player == ConstantWya.P2_MIN))
        point_vals = {k:v for k, v in point_vals.items() if k in empty_blocks}

        non_zero_blocks = set(point_vals.keys())
        minus_vals = {k: sum(v) for k, v in sorted(point_vals.items(), key=lambda item: item[1], reverse=True) if sum(v) < 0}
        point_vals = {k: sum(v) for k, v in sorted(point_vals.items(), key=lambda item: item[1], reverse=True) if sum(v) >= 0}

        actions = []
        if len(point_vals) < ConstantWya.MIN_BRANCH_NUM:
            sub = ConstantWya.MIN_BRANCH_NUM - len(point_vals)
            zero_blocks = list(empty_blocks - non_zero_blocks)
            if len(zero_blocks) < sub: zero_blocks += list(minus_vals.keys())[:sub-len(zero_blocks)+1]
            if len(zero_blocks) >= sub:
                c = random.sample(zero_blocks, sub)
                actions += [(t, 0) for t in c]
        actions += list(point_vals.items()) # NOTE: order matters; action(coordinate): score

        return actions
    
    def next_node_after_action(self, cur_node: StateWya, next_action):
        next_player = ConstantWya.P1_MAX if cur_node.player == ConstantWya.P2_MIN else ConstantWya.P2_MIN
        next_state = cur_node.state.copy()
        next_state[next_action[0][0]][next_action[0][1]] = cur_node.player
        next_h_score = next_action[1]

        return StateWya(next_state, next_player, next_h_score)
    
    def rule_search(self, depth, cur_node: StateWya, alpha = ConstantWya.NEG_INF, beta = ConstantWya.POS_INF):

        best_action = None
        # NOTE: check the search horizontal line
        if depth >= ConstantWya.RULE_DEPTH:
            return (best_action, cur_node.h_score)

        best_score = ConstantWya.NEG_INF
        for a in self.get_child_actions(cur_node):
            val = self.rule_search(depth+1, self.next_node_after_action(cur_node, a), alpha, beta)
            best_score = max(best_score, cur_node.h_score + val[1]/(depth+1))
            if best_score == val[1]: best_action = a

        return (best_action, best_score)
    

    def get_score_after_action(self, grid,x,y):
        score=0

        #1.check horizontal
        if((grid[x][0] != None) and (grid[x][1] != None) and  (grid[x][2]!= None) and (grid[x][3] != None) and (grid[x][4] != None) and (grid[x][5]  != None)):  
            score+=6
        else:
            if (grid[x][0] != None) and (grid[x][1] != None) and  (grid[x][2]!= None) and (grid[x][3] == None):
                if y==0 or y==1 or y==2:
                    score+=3
            elif (grid[x][0] == None) and (grid[x][1] != None) and  (grid[x][2]!= None) and (grid[x][3] != None) and (grid[x][4] == None):
                if y==1 or y==2 or y==3:
                    score+=3
            elif  (grid[x][1] == None) and (grid[x][2] != None) and  (grid[x][3]!= None) and (grid[x][4] != None) and (grid[x][5] == None):
                if y==2 or y==3 or y==4:
                    score+=3
            elif  (grid[x][2] == None) and  (grid[x][3]!= None) and (grid[x][4] != None) and (grid[x][5] != None):
                if y==3 or y==4 or y==5:
                    score+=3
                
        #2.check vertical
        if((grid[0][y] != None) and (grid[1][y] != None) and (grid[2][y] != None) and (grid[3][y] != None) and (grid[4][y]!= None) and (grid[5][y]!= None)):
            score+=6
        else:
            if (grid[0][y] != None) and (grid[1][y] != None) and  (grid[2][y]!= None) and (grid[3][y] == None):
                if x==0 or x==1 or x==2:
                    score+=3
            elif (grid[0][y] == None) and (grid[1][y] != None) and  (grid[2][y]!= None) and (grid[3][y] != None) and (grid[4][y] == None):
                if x==1 or x==2 or x==3:
                    score+=3
            elif (grid[1][y] == None) and (grid[2][y] != None) and  (grid[3][y]!= None) and (grid[4][y] != None) and (grid[5][y] == None):
                if x==2 or x==3 or x==4:
                    score+=3
            elif  (grid[2][y] == None) and  (grid[3][y]!= None) and (grid[4][y] != None) and (grid[5][y] != None):
                if x==3 or x==4 or x==5:
                    score+=3
        return score

    def get_available_actions(self, node: StateWya) -> set:
        empty_blocks = set(t for t in zip(*np.where(node.state == None)) if sum(t)%2 == (node.player == ConstantWya.P2_MIN))
        return empty_blocks
    
    def next_node_after_action_mcts(self, cur_node:StateWya, next_action) -> StateWya:
        """
        next_action(tuple): (x, y)
        """
        next_player = ConstantWya.P1_MAX if cur_node.player == ConstantWya.P2_MIN else ConstantWya.P2_MIN
        next_state = cur_node.state.copy()
        next_state[next_action[0]][next_action[1]] = cur_node.player
        next_h_score = self.get_score_after_action(next_state, next_action[0], next_action[1])

        return StateWya(next_state, next_player, next_h_score)

    def mct_search(self, root_node: StateWya, N=ConstantWya.MCT_N):
        def select(m_node: MCTNodeWya) -> MCTNodeWya:
            if m_node.children: return select(max(m_node.children.keys(), key=ucb))
            return m_node

        def expand(m_node: MCTNodeWya) -> MCTNodeWya:
            if not m_node.children and not root_node.is_terminate():
                m_node.children = {MCTNodeWya(parent=m_node, state=self.next_node_after_action_mcts(m_node.state, a)): a 
                                    for a in self.get_available_actions(m_node.state)
                                    if not self.next_node_after_action_mcts(m_node.state, a).is_terminate()}
            return select(m_node)
        
        def simulate(cur_node: StateWya) -> int:
            player = cur_node.player
            action = ()
            scores = [0, 0] # NOTE: [p1, p2]
            while not cur_node.is_terminate():
                action = random.choice(list(self.get_available_actions(cur_node)))
                cur_node = self.next_node_after_action_mcts(cur_node, action)
                scores[cur_node.player == ConstantWya.P2_MIN] += self.get_score_after_action(cur_node.state, action[0], action[1])
            scores[cur_node.player == ConstantWya.P2_MIN] += self.get_score_after_action(cur_node.state, action[0], action[1])
            return (scores[player == ConstantWya.P2_MIN] - scores[player != ConstantWya.P2_MIN])*2
        
        def backprop(m_node: MCTNodeWya, value):
            m_node.U += value
            m_node.N += 1
            if m_node.parent: backprop(m_node.parent, value)

        root_mct_node = MCTNodeWya(state=root_node)

        for _ in range(N):
            leaf = select(root_mct_node)
            child = expand(leaf)
            value = simulate(child.state)
            backprop(child, value)

        max_state = max(root_mct_node.children, key=lambda x: x.N)
        return list(root_mct_node.children.get(max_state))
    

    def get_move(self, state, player) -> list:
        # NOTE: p1-black-X-even; p2-white-O-odd
        state = np.array(state)
        this_node = StateWya(state, player, 0)

        available_blocks = self.get_available_actions(this_node)

        if len(available_blocks) == 1:
            best_action = ret = list(list(available_blocks)[0])
        elif len(available_blocks) >= 15:
            # print("\n rule \n")
            ret = self.rule_search(0, this_node)
            best_action = list(ret[0][0])
        else:
            # print("\n rule \n")
            ret = self.rule_search(0, this_node)
            best_action = list(ret[0][0])

            if ret[0][1] == 0:
                # print("\n second mct \n")
                best_action2 = ret2 = self.mct_search(this_node)
                best_action = best_action2 if np.random.choice([0, 1], 1, p=[len(available_blocks)/18, 1-len(available_blocks)/18]) else best_action

        return best_action
