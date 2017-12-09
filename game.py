import csv
import pickle
import random
import sys

offensive_play_types = {"quick_two", "quick_three", "best_shot"}
defensive_play_types = {"foul", "best_defense"}
winning_constant = 20
learning = 0.1
discount = 1
team_dict = pickle.load(open("team_dict.pkl", "rb"))
Q = pickle.load(open("Q.pkl", "rb"))

def generate_team_dict():
    team_dict = {}

    with open('team_data.csv') as file:
        reader = csv.DictReader(file)
        for row in reader:
            team_name = row["ï»¿team"]
            value_dict = {}
            for (stat, value) in list(row.items())[1:]:
                value_dict[stat] = float(value)
            team_dict[team_name] = value_dict

    pickle.dump(team_dict, open("team_dict.pkl", "wb"))

class Node(object):
    def __init__(self, children=[], node_type=None, value=None, action=None):
        self.children = children
        self.node_type = node_type
        self.value = value
        self.action = action
        
    def __repr__(self):
        return "<Children: {}, Type: {}, Value: {}, Action: {}>".format(self.children, self.node_type, self.value, self.action)
    
    def __str__(self):
        return "<Children: {}, Type: {}, Value: {}, Action: {}>".format(self.children, self.node_type, self.value, self.action)
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, node, probability=None):
        self.children = self.children + [(node, probability)]
        
def expectiminimax(node, top=False):
    if node.is_leaf():
        return node.value, None
    
    node_value = 0.0
    action = None
    if node.node_type == "max":
        node_value = -sys.maxsize - 1
        for (child, _) in node.children:
            expectiminimax_value = expectiminimax(child)[0]
            if expectiminimax_value > node_value:
                node_value = expectiminimax_value
                action = child.action
    elif node.node_type == "min":
        node_value = sys.maxsize
        for (child, _) in node.children:
            expectiminimax_value = expectiminimax(child)[0]
            if expectiminimax_value < node_value:
                node_value = expectiminimax_value
                action = child.action
    elif node.node_type == "expected":
        for (child, probability) in node.children:
            node_value += expectiminimax(child)[0] * probability
    node.value = node_value
    return node_value, action

class State(object):
    def __init__(self, team_one="", team_two="", possessor="", point_differential=0, time=0):
        self.team_one = team_one
        self.team_two = team_two
        self.possessor = possessor
        self.point_differential = point_differential
        self.time = time
        
    def __repr__(self):
        if self.point_differential == 0:
            return "Tie game, {} seconds left, {} has the ball".format(self.time, self.possessor)
        if self.point_differential > 0:
            winning_team = self.team_one
            losing_team = self.team_two
        else:
            winning_team = self.team_two
            losing_team = self.team_one
        return "{} is up by {}, {} seconds left, {} has the ball".format(
            winning_team, abs(self.point_differential), self.time, self.possessor)
    
    def __str__(self):
        if self.point_differential == 0:
            return "Tie game, {} seconds left, {} has the ball".format(self.time, self.possessor)
        if self.point_differential > 0:
            winning_team = self.team_one
            losing_team = self.team_two
        else:
            winning_team = self.team_two
            losing_team = self.team_one
        return "{} is up by {}, {} seconds left, {} has the ball".format(
            winning_team, abs(self.point_differential), self.time, self.possessor)

def generate_game_tree(state, n_type, action_type=None):  
    if state.time < 3:
        game_value = winning_constant + state.point_differential if state.point_differential > 0 else state.point_differential-winning_constant if state.point_differential < 0 else 0
        return Node(value=game_value)
    current_node = Node(node_type=n_type, action=action_type)
    if n_type == "max" and state.possessor == state.team_one:
        for offensive_play in offensive_play_types:
            current_node.add_child(generate_game_tree(state, n_type="expected", action_type=offensive_play))
    elif n_type == "max":
        for defensive_play in defensive_play_types:
            current_node.add_child(generate_game_tree(state, n_type="expected", action_type=defensive_play))
    elif n_type == "expected":
        possessor = state.possessor
        not_possessor = state.team_one if state.team_two == possessor else state.team_two
        if action_type == "quick_two" or action_type == "quick_three":
            #turnover
            time_taken = 5 if state.time > 5 else state.time
            turnover_prob = team_dict[possessor]["to%"]/100.0
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), turnover_prob)
            
            #fouled
            points_scored = 2 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"] if action_type == "quick_two" else 3 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]
            foul_prob = (team_dict[possessor]["o_foul%"] + team_dict[not_possessor]["d_foul%"])/200.0
            probability = (1 - turnover_prob) * foul_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential + points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #makes shot
            points_scored = 2 if action_type == "quick_two" else 3
            two_prob = (((team_dict[possessor]["o_fgm"] - team_dict[possessor]["o_3pm"]) / (team_dict[possessor]["o_fga"] - team_dict[possessor]["o_3pa"])) 
                        + ((team_dict[not_possessor]["d_fgm"] - team_dict[not_possessor]["d_3pm"]) / (team_dict[not_possessor]["d_fga"] - team_dict[not_possessor]["d_3pa"])))/2.0
            three_prob = ((team_dict[possessor]["o_3pm"] / team_dict[possessor]["o_3pa"]) 
                            + (team_dict[not_possessor]["d_3pm"] / team_dict[not_possessor]["d_3pa"]))/2.0
            if action_type == "quick_two":
                points_scored = 2
                probability = (1 - turnover_prob) * (1 - foul_prob) * two_prob
            else:
                points_scored = 3
                probability = (1 - turnover_prob) * (1 - foul_prob) * three_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential + points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #misses shot, opponent gets rebound
            def_reb_prob = ((100.0 - team_dict[possessor]["or%"]) + team_dict[possessor]["dr%"])/200.0
            if action_type == "quick_two":
                probability = (1 - turnover_prob) * (1 - foul_prob) * (1 - two_prob) * def_reb_prob
            else:
                probability = (1 - turnover_prob) * (1 - foul_prob) * (1 - three_prob) * def_reb_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #misses shot, team gets offensive rebound
            if action_type == "quick_two":
                probability = (1 - turnover_prob) * (1 - foul_prob) * (1 - two_prob) * (1 - def_reb_prob) 
            else:
                probability = (1 - turnover_prob) * (1 - foul_prob) * (1 - three_prob) * (1 - def_reb_prob)
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
        elif action_type == "best_shot":
            #turnover
            time_taken = 24 if state.time > 24 else state.time
            turnover_prob = team_dict[possessor]["to%"]/100.0
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), turnover_prob)
            
            #fouled
            points_scored = 2 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]
            foul_prob = (team_dict[possessor]["o_foul%"] + team_dict[not_possessor]["d_foul%"])/200.0
            probability = (1.0 - turnover_prob) * foul_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential + points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #makes two
            points_scored = 2
            takes_three_prob = ((team_dict[possessor]["o_3pa"]/team_dict[possessor]["o_fga"]) 
                                + (team_dict[possessor]["o_3pa"]/team_dict[possessor]["o_fga"]))/2
            two_prob = (((team_dict[possessor]["o_fgm"] - team_dict[possessor]["o_3pm"]) / (team_dict[possessor]["o_fga"] - team_dict[possessor]["o_3pa"])) 
                        + ((team_dict[not_possessor]["d_fgm"] - team_dict[not_possessor]["d_3pm"]) / (team_dict[not_possessor]["d_fga"] - team_dict[not_possessor]["d_3pa"])))/2.0
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * (1.0 - takes_three_prob) * two_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential + points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #makes three
            points_scored = 3
            three_prob = ((team_dict[possessor]["o_3pm"] / team_dict[possessor]["o_3pa"]) 
                            + (team_dict[not_possessor]["d_3pm"] / team_dict[not_possessor]["d_3pa"]))/2.0
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * takes_three_prob * three_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential + points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #misses shot, opponent gets rebound
            miss_prob = 1.0 - ((team_dict[possessor]["o_fgm"] / team_dict[possessor]["o_fga"]) 
                            + (team_dict[not_possessor]["d_fgm"] / team_dict[not_possessor]["d_fga"]))/2.0
            def_reb_prob = ((100.0 - team_dict[possessor]["or%"]) + team_dict[possessor]["dr%"])/200.0
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * miss_prob * def_reb_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #misses shot, team gets offensive rebound
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * miss_prob * (1.0 - def_reb_prob)
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
        elif action_type == "foul":
            #turnover
            time_taken = 5 if state.time > 5 else state.time
            probability = team_dict[possessor]["to%"]/100.0
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #fouled
            points_scored = 2.0 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]
            probability= 1.0 - (team_dict[possessor]["to%"]/100.0)
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential - points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
        elif action_type == "best_defense":
            #turnover
            time_taken = 20 if state.time > 20 else state.time
            turnover_prob = team_dict[possessor]["to%"]/100.0
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), turnover_prob)
            
            #fouled
            points_scored = 2.0 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]
            foul_prob = (team_dict[possessor]["o_foul%"] + team_dict[not_possessor]["d_foul%"])/200.0
            probability = (1.0 - turnover_prob) * foul_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential - points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #makes two
            points_scored = 2
            takes_three_prob = ((team_dict[possessor]["o_3pa"]/team_dict[possessor]["o_fga"]) 
                                + (team_dict[possessor]["o_3pa"]/team_dict[possessor]["o_fga"]))/2
            two_prob = (((team_dict[possessor]["o_fgm"] - team_dict[possessor]["o_3pm"]) / (team_dict[possessor]["o_fga"] - team_dict[possessor]["o_3pa"])) 
                        + ((team_dict[not_possessor]["d_fgm"] - team_dict[not_possessor]["d_3pm"]) / (team_dict[not_possessor]["d_fga"] - team_dict[not_possessor]["d_3pa"])))/2.0
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * (1.0 - takes_three_prob) * two_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential - points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #makes three
            points_scored = 3
            three_prob = ((team_dict[possessor]["o_3pm"] / team_dict[possessor]["o_3pa"]) 
                            + (team_dict[not_possessor]["d_3pm"] / team_dict[not_possessor]["d_3pa"]))/2.0
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * takes_three_prob * three_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential - points_scored, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #misses shot, opponent gets rebound
            miss_prob = 1.0 - ((team_dict[possessor]["o_fgm"] / team_dict[possessor]["o_fga"]) 
                            + (team_dict[not_possessor]["d_fgm"] / team_dict[not_possessor]["d_fga"]))/2.0
            def_reb_prob = ((100.0 - team_dict[possessor]["or%"]) + team_dict[not_possessor]["dr%"])/200.0
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * miss_prob * def_reb_prob
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = not_possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
            
            #misses shot, team gets offensive rebound
            probability = (1.0 - turnover_prob) * (1.0 - foul_prob) * miss_prob * (1.0 - def_reb_prob)
            new_state = State(team_one = state.team_one, 
                              team_two = state.team_two, 
                              possessor = possessor, 
                              point_differential = state.point_differential, 
                              time = state.time - time_taken)
            current_node.add_child(generate_game_tree(new_state, "max"), probability)
    if current_node.is_leaf():
        game_value = winning_constant + state.point_differential if state.point_differential > 0 else state.point_differential-winning_constant if state.point_differential < 0 else 0
        return Node(value=game_value)
    return current_node

def find_optimal_play(state, p):
    game_tree = generate_game_tree(state, "max")
    value, minimax_action = expectiminimax(game_tree, True)
    
    q_learning_actions = []
    max_value = -sys.maxsize - 1
    if state.possessor == state.team_one:
        moves = offensive_play_types
    else:
        moves = defensive_play_types
    for move in moves:
        q_value = Q[(state.point_differential, state.time, move)] if (state.point_differential, state.time, move) in Q else 0
        if q_value >= max_value:
            q_learning_actions.append(move)
            max_value = q_value
    if minimax_action in q_learning_actions:
        final_action = minimax_action
    else:
        final_action = random.choice([minimax_action] + q_learning_actions)
    if p:
        print("Expectiminimax decision: {}, q-learning decision(s): {}, final decision: {}".format(minimax_action, q_learning_actions, final_action))
    return final_action

def update_Q(state, action, outcome, time_taken):
    if state.time < 3:
        for play in offensive_play_types | defensive_play_types:
            Q[(state.point_differential, state.time, play)] = state.point_differential if state.point_differential == 0 else state.point_differential + winning_constant if state.point_differential > 0 else state.point_differential - winning_constant
        return 
    if action in offensive_play_types:
        next_actions = defensive_play_types
    else:
        next_actions = offensive_play_types
    possessor = state.possessor
    not_possessor = state.team_one if possessor == state.team_two else state.team_two
    new_state = State(state.team_one,
                      state.team_two, 
                      not_possessor, 
                      state.point_differential + outcome, 
                      max(state.time - time_taken, 0))
    valid_actions = []
    for next_action in next_actions:
        if (state.point_differential + outcome, max(state.time - time_taken, 0), next_action) in Q:
            valid_actions.append(next_action)
    if (state.point_differential, state.time, action) in Q:
        if valid_actions:
            Q[(state.point_differential, state.time, action)] = (1.0 - learning) * Q[(state.point_differential, state.time, action)] + \
                learning * (outcome + discount * max([Q[(state.point_differential + outcome, max(state.time - time_taken, 0), act)] for act in valid_actions]))
        else:
            Q[(state.point_differential, state.time, action)] = (1.0 - learning) * Q[(state.point_differential, state.time, action)] + learning * outcome
    else:
        if valid_actions:
            Q[(state.point_differential, state.time, action)] = learning * (outcome + discount * max([Q[(state.point_differential + outcome, max(state.time - time_taken, 0), act)] for act in valid_actions]))
        else:
            Q[(state.point_differential, state.time, action)] = learning * outcome

def execute_play(state, your_play, opponent_play, p):
    possessor = state.possessor
    not_possessor = state.team_one if state.team_two == possessor else state.team_two
    
    points_scored = 0
    time_taken = 0
    new_possessor = possessor
    if your_play == "best_defense" or opponent_play == "best_defense":
        if your_play == "quick_two" or opponent_play == "quick_two":
            time_taken = random.randint(3, 5) if state.time > 5 else state.time
            #check for turnover
            if random.randint(0, 1000) <= 10 * team_dict[possessor]["to%"]:
                new_possessor = not_possessor
                if p:
                    print("{} turns the ball over".format(possessor))
            else:
                #team gets fouled
                if random.randint(0, 1000) <= 5 * (team_dict[possessor]["o_foul%"] + team_dict[not_possessor]["d_foul%"]):
                    new_possessor = not_possessor
                    for _ in range(2):
                        if random.randint(0, 1000) <= 1000 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]:
                            if your_play == "quick_two":
                                points_scored += 1
                            else:
                                points_scored -= 1
                    if p:
                        print("{} gets fouled and makes {} free throws".format(possessor, abs(points_scored)))
                else:
                    #team makes a two
                    if random.randint(0, 1000) <= 500 * (((team_dict[possessor]["o_fgm"] - team_dict[possessor]["o_3pm"]) / (team_dict[possessor]["o_fga"] - team_dict[possessor]["o_3pa"])) 
                                                         + ((team_dict[not_possessor]["d_fgm"] - team_dict[not_possessor]["d_3pm"]) / (team_dict[not_possessor]["d_fga"] - team_dict[not_possessor]["d_3pa"]))):
                        if your_play == "quick_two":
                            points_scored += 2
                        else:
                            points_scored -= 2
                        new_possessor = not_possessor
                        if p:
                            print("{} makes the quick two".format(possessor))
                    #if missed, check for offensive rebound
                    if points_scored == 0:
                        if random.randint(0, 1000) > 5 * (team_dict[possessor]["or%"] + (100.0 - team_dict[not_possessor]["dr%"])):
                            new_possessor = not_possessor
                            if p:
                                print("{} misses the quick two and {} gets the rebound".format(possessor, not_possessor))
                        else:
                            if p:
                                print("{} misses the quick two and gets an offensive rebound".format(possessor))
        elif your_play == "quick_three" or opponent_play == "quick_three":
            time_taken = random.randint(3, 5) if state.time > 5 else state.time
            #check for turnover
            if random.randint(0, 1000) <= 10 * team_dict[possessor]["to%"]:
                new_possessor = not_possessor
                if p:
                    print("{} turns the ball over".format(possessor))
            else:
                #team gets fouled
                if random.randint(0, 1000) <= 2.5 * (team_dict[possessor]["o_foul%"] + team_dict[not_possessor]["d_foul%"]):
                    new_possessor = not_possessor
                    for _ in range(3):
                        if random.randint(0, 1000) <= 1000 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]:
                            if your_play == "quick_three":
                                points_scored += 1
                            else:
                                points_scored -= 1
                    if p:
                        print("{} gets fouled and makes {} free throws".format(possessor, abs(points_scored)))
                else:
                    #team makes a three
                    if random.randint(0, 1000) <= 500 * ((team_dict[possessor]["o_3pm"] / team_dict[possessor]["o_3pa"]) 
                                                         + (team_dict[not_possessor]["d_3pm"] / team_dict[not_possessor]["d_3pa"])):
                        if your_play == "quick_three":
                            points_scored += 3
                        else:
                            points_scored -= 3
                        new_possessor = not_possessor
                        if p:
                            print("{} makes the quick three".format(possessor))
                    #if missed, check for offensive rebound
                    if points_scored == 0:
                        if random.randint(0, 1000) > 5 * (team_dict[possessor]["or%"] + (100.0 - team_dict[not_possessor]["dr%"])):
                            new_possessor = not_possessor
                            if p:
                                print("{} misses the quick three and {} gets the rebound".format(possessor, not_possessor))
                        else:
                            if p:
                                print("{} misses the quick three and gets an offensive rebound".format(possessor))
        elif your_play == "best_shot" or opponent_play == "best_shot":
            #team turns over the ball
            if random.randint(0, 1000) <= 10 * team_dict[possessor]["to%"]:
                time_taken = random.randint(5, 22)
                new_possessor = not_possessor
                if p:
                    print("{} turns the ball over".format(possessor))
            else:
                time_taken = random.randint(15, 24)
                #team gets fouled
                if random.randint(0, 1000) <= 5 * (team_dict[possessor]["o_foul%"] + team_dict[not_possessor]["d_foul%"]):
                    new_possessor = not_possessor
                    for _ in range(2):
                        if random.randint(0, 1000) <= 1000 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]:
                            if your_play == "best_shot":
                                points_scored += 1
                            else:
                                points_scored -= 1
                    if p:
                        print("{} gets fouled and makes {} free throws".format(possessor, abs(points_scored)))
                else:
                    #opponent takes a three
                    if random.randint(0, 1000) <= 500 * ((team_dict[possessor]["o_3pa"] / team_dict[possessor]["o_fga"])
                                                        + (team_dict[not_possessor]["d_3pa"] / team_dict[not_possessor]["d_fga"])):
                        #team makes a three
                        if random.randint(0, 1000) <= 500 * ((team_dict[possessor]["o_3pm"] / team_dict[possessor]["o_3pa"])
                                                        + (team_dict[not_possessor]["d_3pm"] / team_dict[not_possessor]["d_3pa"])):
                            new_possessor = not_possessor
                            if your_play == "best_shot":
                                points_scored += 3
                            else:
                                points_scored -= 3
                            if p:
                                print("{} makes a three".format(possessor))
                        else:
                            if p:
                                print("{} misses a three".format(possessor))
                    else:
                        #team makes a two
                        if random.randint(0, 1000) <= 500 * (((team_dict[possessor]["o_fgm"] - team_dict[possessor]["o_3pm"]) / (team_dict[possessor]["o_fga"] - team_dict[possessor]["o_3pa"]))
                                                        + ((team_dict[not_possessor]["d_fgm"] - team_dict[not_possessor]["d_3pm"]) / (team_dict[not_possessor]["d_fga"] - team_dict[not_possessor]["d_3pa"]))):
                            new_possessor = not_possessor
                            if your_play == "best_shot":
                                points_scored += 2
                            else:
                                points_scored -= 2
                            if p:
                                print("{} makes a two".format(possessor))
                        else:
                            if p:
                                print("{} misses a two".format(possessor))
                    #if team missed, check for offensive rebound
                    if points_scored == 0:
                        if random.randint(0, 1000) > 5 * (team_dict[possessor]["or%"] + (100.0 - team_dict[not_possessor]["dr%"])):
                            new_possessor = not_possessor
                        else:
                            if p:
                                print("{} misses the shot but gets an offensive rebound".format(possessor))
    elif your_play == "foul" or opponent_play == "foul":
        #opponent turns over the ball
        if random.randint(0, 1000) <= 10*team_dict[possessor]["to%"]:
            time_taken = random.randint(1, 3)
            new_possessor = not_possessor
            if p:
                print("{} turns the ball over".format(possessor))
        #foul happens
        else:
            time_taken = random.randint(1, 3)
            new_possessor = not_possessor
            for _ in range(2):
                if random.randint(0, 1000) <= 1000 * team_dict[possessor]["o_ftm"] / team_dict[possessor]["o_fta"]:
                    if your_play == "foul":
                        points_scored -= 1
                    else:
                        points_scored += 1
            if p:
                print("{} is intentionally fouled and makes {} free throws".format(possessor, abs(points_scored)))
    return State(team_one = state.team_one, 
                 team_two = state.team_two, 
                 possessor = new_possessor, 
                 point_differential = state.point_differential + points_scored, 
                 time = max(state.time - time_taken, 0))

def main(p=True):
    random.seed(3)
    team_one = input("Enter your team: ")
    while team_one not in team_dict:
    	team_one = input("Invalid team, enter a valid team name: ")
    team_two = input("Enter opponent's team: ")
    while team_two not in team_dict:
    	team_two = input("Invalid team, enter a valid team name: ")
    if p:
        print("Your Team: {}, Opponent: {}".format(team_one, team_two))

    point_differential = int(input("Enter a point differential between 5 and -5 (positive means you are winning): "))

    time = int(input("Enter the amount of time left between 0 and 30: "))

    if point_differential >= 0:
        winning_team = team_one
        losing_team = team_two
    else:
        winning_team = team_two
        losing_team = team_one
    your_state = State(team_one, team_two, losing_team, point_differential, time)
    opponent_state = State(team_two, team_one, losing_team, -point_differential, time)
    
    while your_state.time > 0:
        if p:
            print()
            print(your_state)
        your_play = find_optimal_play(your_state, p)
        opponent_play = find_optimal_play(opponent_state, p)
        if your_play is None or opponent_play is None:
            break
        your_temp_state = execute_play(your_state, your_play, opponent_play, p)
        update_Q(your_state, your_play, your_temp_state.point_differential - your_state.point_differential, your_state.time - your_temp_state.time)
        your_state = your_temp_state
        opponent_state = State(team_two, team_one, your_state.possessor, -your_state.point_differential, your_state.time)
    if your_state.point_differential > 0 and p:
        print("Game is over, you win by {}".format(your_state.point_differential))
    elif your_state.point_differential < 0 and p:
        print("Game is over, you lose by {}".format(abs(your_state.point_differential)))
    elif p:
        print("Game is over, tied")
    pickle.dump(Q, open("Q.pkl", "wb"))

if __name__ == "__main__":
    main()