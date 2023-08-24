import time

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import py3langid
import textstat
from langdetect import detect
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from webdriver_manager.firefox import GeckoDriverManager
from wordcloud import WordCloud


def whatsapp_open(username, range_scroll):
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))
    driver.get("https://web.whatsapp.com/")

    try:
        # Wait for the QR code element to appear within 40 seconds
        WebDriverWait(driver, 40).until(
            EC.presence_of_element_located((By.CLASS_NAME, "_10aH-"))
        )
        print("QR code scanned successfully!")

    except:
        # If the QR code element is not found within 40 seconds, stop with an error message
        print("Error: Unable to scan QR code within 40 seconds!")
        driver.quit()

    message = "Scan QR Code To Login"
    driver.execute_script(f"alert('{message}')")
    time.sleep(75)

    element = driver.find_element(
        By.CLASS_NAME, "to2l77zo.gfz4du6o.ag5g9lrv.bze30y65.kao4egtt.qh0vvdkp"
    )
    name = username

    time.sleep(5)

    for word in name:
        element.clear()  # Clear the previous input
        element.send_keys(word)

    found = driver.find_element(By.CLASS_NAME, "matched-text._11JPr")
    found.click()

    def scroll():  # noqa: E999
        # Scroll up by pressing the Page Up key
        from selenium.webdriver.common.keys import Keys

        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.PAGE_UP)

        # Alternatively, you can scroll by executing JavaScript code
        driver.execute_script("window.scrollBy(0, -window.innerHeight);")

    for i in range(range_scroll):
        time.sleep(5)
        scroll()

    time.sleep(10)
    chats = driver.find_elements(By.CLASS_NAME, "_21Ahp")
    for i in chats:
        print(i.text)

    time.sleep(5)
    chat = []
    for i in chats:
        chat.append(i.text)

    df = pd.DataFrame({"Chatting": chat})
    df.to_csv("Friend_chatting_new.csv")

    time.sleep(3)

    pattern = r"(https?://\S+)"
    df["link_count"] = df["Chatting"].str.count(pattern)
    df["word_count"] = df["Chatting"].apply(lambda x: len(x.split()))
    df["sentence_length"] = df["Chatting"].apply(lambda x: len(x))
    df["has_numeric"] = df["Chatting"].apply(
        lambda x: any(char.isdigit() for char in x)
    )
    emoticons = [":)", ":(", ";)", ":D", ":P"]

    df["has_emoticon"] = df["Chatting"].apply(
        lambda x: any(emoticon in x for emoticon in emoticons)
    )

    # Define a function for language detection
    def detect_language(text):
        # Preprocess Romanized Urdu text
        if any(ord(char) > 128 for char in text):
            # Replace Urdu characters with empty string
            text = "".join(char for char in text if ord(char) <= 128)

        try:
            return detect(text)
        except:
            return "Unknown"

    # Apply language detection to the 'Chatting' column
    df["language_detection"] = df["Chatting"].apply(detect_language)

    def detect_language(text):
        lang, confidence = py3langid.classify(text)
        return lang, confidence

    df[["language_py3_detection", "py3_confidence"]] = (
        df["Chatting"].apply(detect_language).apply(pd.Series)
    )

    nltk.download("stopwords")
    # Stopword Count
    stopwords_set = set(stopwords.words("english"))
    df["stopword_count"] = df["Chatting"].apply(
        lambda x: len([w for w in x.lower().split() if w in stopwords_set])
    )

    df["repeated_chars_count"] = df["Chatting"].apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i] == x[i + 1])
    )

    nltk.download("vader_lexicon")
    sid = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["Chatting"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )

    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")

    def count_pos_tags(text):
        pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
        counts = {}
        for _, tag in pos_tags:
            counts[tag] = counts.get(tag, 0) + 1
            return counts

    df["pos_tag_counts"] = df["Chatting"].apply(count_pos_tags)
    df["flesch_reading_ease"] = df["Chatting"].apply(textstat.flesch_reading_ease)
    df["gunning_fog_index"] = df["Chatting"].apply(textstat.gunning_fog)

    df.to_csv("Feature_Engineering.csv")

    # Create an instance of CountVectorizer
    vectorizer_1 = CountVectorizer()

    # Apply count vectorization on the 'Chatting' column
    count_vectors = vectorizer_1.fit_transform(df["Chatting"])

    # Convert the count vectors to a DataFrame
    count_vector_df = pd.DataFrame(
        count_vectors.toarray(), columns=vectorizer_1.get_feature_names_out()
    )

    count_vector_df.to_csv("after_count_vectorizer.csv")

    # Create an instance of TfidfVectorizer
    vectorizer_2 = TfidfVectorizer()

    # Apply TF-IDF vectorization on the 'Chatting' column
    tfidf_vectors = vectorizer_2.fit_transform(df["Chatting"])

    # Convert the TF-IDF vectors to a DataFrame
    tfidf_vector_df = pd.DataFrame(
        tfidf_vectors.toarray(), columns=vectorizer_2.get_feature_names_out()
    )
    tfidf_vector_df.to_csv("after_tfidf_vectorizer.csv")

    # Get the vocabulary of CountVectorizer
    count_vocab = vectorizer_1.vocabulary_

    # Count the occurrences of each word in CountVectorizer
    count_word_counts = {
        word: count_vectors[:, index].sum() for word, index in count_vocab.items()
    }

    # Get the vocabulary of TfidfVectorizer
    tfidf_vocab = vectorizer_2.vocabulary_

    # Count the occurrences of each word in TfidfVectorizer
    tfidf_word_counts = {
        word: tfidf_vectors[:, index].sum() for word, index in tfidf_vocab.items()
    }

    # Print the word counts for CountVectorizer
    print("Count Vectorizer Word Counts:")
    for word, count in count_word_counts.items():
        print(f"{word}: {count}")

    # Print the word counts for TfidfVectorizer
    print("\nTF-IDF Vectorizer Word Counts:")
    for word, count in tfidf_word_counts.items():
        print(f"{word}: {count}")

    text = " ".join(df["Chatting"])

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )

    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.savefig("wordcloud.png")
    plt.axis("off")
    plt.show()


# whatsapp_open("abdullah home", 10)
