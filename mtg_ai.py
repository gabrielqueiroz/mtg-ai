from transformers import pipeline
import csv
import requests

def read_csv_to_array(file_path):
    data_array = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data_array.append(row)
        return data_array
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except csv.Error as e:
        print(f"CSV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def get_card_text(card_name):
    encoded_name = requests.utils.quote(card_name)
    url = f"https://api.scryfall.com/cards/named?exact={encoded_name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        card_data = response.json()
        return card_data.get('oracle_text', 'Oracle text not found.')
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    except ValueError:
        return "Error: Unable to parse the API response."

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def categorize_card(card_text):
    categories = [
        "Ramp: Increases the amount of available mana for a given turn. Examples: Playing additional lands, putting lands on battlefield, tapping for mana.",
        "Board Wipe: Destroys all creatures and/or permanents on the battlefield",
        "Removal: Remove a permanent that the opponent controls. Examples: Destroy target, exile target, return target permanent to its owner's hand.",
        "Win Condition: Ways for the player to win the game or cause the opponent to lose. Examples: you win the game, delas 50 damage to target player"
    ]
    result = classifier(card_text, categories)
    categories = [label for label, score in zip(result['labels'], result['scores']) if score > 0.5]
    categories = [label.split(':')[0] for label in categories]
    return categories

cards = read_csv_to_array('sample.csv')
for card in cards:
    card_text = get_card_text(card[0])
    categories = categorize_card(card_text)
    print(f"{card[0]}: {categories}")