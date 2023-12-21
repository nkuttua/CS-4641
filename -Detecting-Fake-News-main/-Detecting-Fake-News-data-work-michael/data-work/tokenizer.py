import csv
import string
import nltk

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def tokenize_text(text, stop_words):

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokenized_lower = tokenizer.tokenize(text.lower())

    filtered_stopwords = [w for w in tokenized_lower if not w in stop_words] 

    non_ascii_removed = [w for w in filtered_stopwords if is_ascii(w)]

    punctuation_removed = [w for w in non_ascii_removed if w.strip(string.punctuation) != '']

    return punctuation_removed

def read_write_csv(data_file, output):
    valid_rows = []
    i = 0
    # id, title, author, text, label
    with open(data_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            printable_text = tokenize_text(row.get("text"), stop_words)
            if len(printable_text) > 50:
                row.update(id=i, text=" ".join(printable_text))
                valid_rows.append(row)
                i += 1

    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['id', 'title', 'author', 'text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)

if __name__ == "__main__":
    # download stuff for nltk
    nltk.download('punkt', download_dir="./env/nltk_data/")
    nltk.download('stopwords', download_dir="./env/nltk_data/")

    stop_words = set(nltk.corpus.stopwords.words('english'))

    train_file = "kaggle-training-fake-news.csv"
    
    read_write_csv(train_file, "train.csv")
