import random
#ansi colors for terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
GREY = '\033[90m'
RESET = '\033[0m'
WORDS=open("word_list.txt").read().splitlines() #creates a list of all 5 letter words
word = random.choice(WORDS) #randomly selects a word from the list
attempts = 6
print("Welcome to Wordle! Guess the 5-letter word. If a letter is in the correct position, it will be shown in green. If it's in the word but in the wrong position, it will be shown in yellow. If it's not in the word, it will be shown in grey.")
for i in range(attempts):
    guess = input(f"Attempt {i+1}/{attempts}: ").lower()
    if len(guess) != 5:
        print("Please enter a 5-letter word.")
        continue
    result = ""
    word_chars = list(word) #converting into a list 
    guess_chars = list(guess)
    marks = [''] * 5
    for i in range(5): #finding green letters 
        if guess_chars[i] == word_chars[i]:
            marks[i] = GREEN
            word_chars[i] = None
    for i in range(5): #finding yellow and grey letters
        if marks[i] == '':
            if guess_chars[i] in word_chars:
                marks[i] = YELLOW
                word_chars[word_chars.index(guess_chars[i])] = None
            else:
                marks[i] = GREY
    for i in range(5): #creating the color coded guessed string
        result += f"{marks[i]}{guess_chars[i].upper()}{RESET}"
    print(result)
    if guess == word:
        print(f"{GREEN}You win! You guessed the word!{RESET}")
        break
else:
    print(f"{GREY}Not so fortunately...the word was: {word.upper()}{RESET}")