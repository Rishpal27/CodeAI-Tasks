import random
GREEN = '\033[92m'
YELLOW = '\033[93m'
GREY = '\033[90m'
RESET = '\033[0m'
WORDS = open("word_list.txt").read().splitlines()
word = random.choice(WORDS)
attempts = 6
print("Welcome to Wordle! Guess the 5-letter word. \n"
"If a letter is in the correct position, " \
"it will be shown in green.\n " \
"If it's in the word but in the wrong position, " \
"it will be shown in yellow. \n" \
"If it's not in the word, it will be shown in grey.")

attempt = 0
while attempt < attempts:
    guess = input(f"Attempt {attempt+1}/{attempts}: ").lower()
    if len(guess) != 5:
        print("Please enter a 5-letter word.")
        continue
    if guess not in WORDS:
        print("Word not in the dictionary. Please try again.")
        continue
    result = ""
    word_chars = list(word)
    guess_chars = list(guess)
    marks = [''] * 5
    for i in range(5):
        if guess_chars[i] == word_chars[i]:
            marks[i] = GREEN
            word_chars[i] = None
    for i in range(5):
        if marks[i] == '':
            if guess_chars[i] in word_chars:
                marks[i] = YELLOW
                word_chars[word_chars.index(guess_chars[i])] = None
            else:
                marks[i] = GREY
    for i in range(5):
        result += f"{marks[i]}{guess_chars[i].upper()}{RESET}"
    print(result)
    if guess == word:
        print(f"{GREEN}You win! You guessed the word!{RESET}")
        break
    attempt += 1
else:
    print(f"{GREY}Not so fortunately...the word was: {word.upper()}{RESET}")