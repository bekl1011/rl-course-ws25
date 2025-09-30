import random
from collections import defaultdict

# ---------- Global game state ----------
game_state = []
N = 3  # board size, default
stats = defaultdict(lambda: [0,0])  # Monte Carlo stats

# ---------- Helpers ----------
def init_game():
    """Initialize global game state and size."""
    global game_state, N, stats
    game_state = ["_"] * (N * N)
    stats = defaultdict(lambda: [0,0])  # reset stats

def reset_game():
    global game_state
    game_state = ["_"] * (N * N)

def print_board():
    for i in range(N):
        print(game_state[i*N:(i+1)*N])

def available_actions():
    return [i for i,v in enumerate(game_state) if v == "_"]

def update_board(pos, symbol):
    global game_state
    if game_state[pos] != "_":
        print("Position already taken!")
    game_state[pos] = symbol

def check_winner():
    # Write logic to decide if game is finished
    # and print the result ("X", "O", "draw")
    return None

# ---------- Monte Carlo Training ----------
def train_monte_carlo_vs_random(episodes=5000):
    """Train Monte Carlo agent as X vs random O."""
    for _ in range(episodes):
        reset_game()
        moves = []
        current = "x"
        while True:
            if current == "x":
                a = random.choice(available_actions())
                moves.append((tuple(game_state), a, current))
                update_board(a, "x")
            else:
                a = random.choice(available_actions())
                update_board(a, "o")
            winner = check_winner()
            if winner:
                reward = 1 if winner == "x" else 0 if winner == "draw" else -1
                for state, act, player in moves:
                    if player == "x":
                        stats[(state, act, player)][0] += reward
                        stats[(state, act, player)][1] += 1
                break
            current = "o" if current == "x" else "x"

def train_monte_carlo_selfplay(episodes=5000):
    """Train Monte Carlo agent with self-play (both X and O learn)."""
    for _ in range(episodes):
        reset_game()
        moves = []
        current = "x"
        while True:
            acts = available_actions()
            # pick greedy move if seen, else random
            best_score, best_action = -1, None
            for a in acts:
                wins_, plays = stats[(tuple(game_state), a, current)]
                score = wins_ / plays if plays > 0 else 0.0
                if score > best_score:
                    best_score, best_action = score, a
            action = best_action if best_action is not None else random.choice(acts)

            moves.append((tuple(game_state), action, current))
            update_board(action, current)

            winner = check_winner()
            if winner:
                for state, act, player in moves:
                    if winner == "draw":
                        reward = 0
                    elif winner == player:
                        reward = 1
                    else:
                        reward = -1
                    stats[(state, act, player)][0] += reward
                    stats[(state, act, player)][1] += 1
                break
            current = "o" if current == "x" else "x"

# ---------- Agents ----------
def random_agent(_symbol="x"):
    return random.choice(available_actions())

def monte_carlo_agent(symbol="x"):
    acts = available_actions()
    best_score, best_action = -1, None
    for a in acts:
        wins_, plays = stats[(tuple(game_state), a, symbol)]
        score = wins_ / plays if plays > 0 else 0.0
        if score > best_score:
            best_score, best_action = score, a
    return best_action if best_action is not None else random.choice(acts)

# ---------- Play against human ----------
def play_vs_human(agent):
    reset_game()
    current = choose_starter()
    print(f"Positions are numbered 0 to {N*N-1}.")
    for i in range(N):
        print([j for j in range(i * N, (i + 1) * N)])
    while True:
        print_board()
        if current == "x":
            a = agent("x")
            print(f"Agent plays: {a}")
            update_board(a, "x")
        else:
            acts = available_actions()
            a = None
            while a not in acts:
                try:
                    a = int(input(f"Your turn (o). Available {acts}: "))
                except ValueError:
                    a = None
            update_board(a, "o")
        winner = check_winner()
        if winner:
            print_board()
            print("Result:", winner)
            return
        current = "o" if current == "x" else "x"

def choose_starter():
    print("Please choose, who starts the game!")
    print("1. Agent")
    print("2. You")
    choice = input("Choose [1/2]: ")
    while choice not in ["1","2"]:
        print("Please give a correct input!")
        choice = input("Choose [1/2]: ")
    return "x" if choice == "1" else "o"

# ---------- Main ----------
if __name__ == "__main__":
    print("Welcome to TicTacToe!")
    #Ask for game Board size and initialize board accordingly
    init_game()

    print("1. Play against Random Agent")
    print("2. Train Monte Carlo Agent vs Random and play")
    print("3. Train Monte Carlo Agent with Self-Play and play")
    choice = input("Choose [1/2/3]: ")
    while choice not in ["1", "2", "3"]:
        print("Please give a correct input!")
        choice = input("Choose [1/2/3]: ")

    if choice == "2":
        print("Training Monte Carlo Agent vs Random…")
        train_monte_carlo_vs_random(episodes=20000)
        print("Done!")
    elif choice == "3":
        print("Training Monte Carlo Agent with Self-Play…")
        train_monte_carlo_selfplay(episodes=20000)
        print("Done!")

    agent = random_agent if choice == "1" else monte_carlo_agent

    print("Play against the agent! (Press Ctrl+C to quit)")
    try:
        while True:
            play_vs_human(agent)
    except KeyboardInterrupt:
        print("\nThanks for playing! Bye 👋")
