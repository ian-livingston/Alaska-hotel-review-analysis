import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn import decomposition
from nltk import word_tokenize, FreqDist
import re
import nltk
from nltk import stem
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD, NMF
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pkg_resources
from symspellpy import SymSpell, Verbosity
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import en_core_web_sm
nlp = en_core_web_sm.load()
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from urllib.request import urlopen
import json
import spacy
from collections import Counter
from wordcloud import WordCloud, get_single_color_func
from collections import defaultdict



# For scraping Tripadvisor URLs for all listed Alaska properties
def get_alaska_hotel_urls():
    '''Takes no arguments and returns a list of URLs to use in scraping reviews.
    Arguments: (none)
    Returns: list
    '''

    urls = []
    hotels = []

    driver = webdriver.Chrome(chromedriver)
    driver.get("https://www.tripadvisor.com/Hotels-g28923-Alaska-Hotels.html")
    hotels_toggle = driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div/div[1]/div[1]/div[5]/div/div/div[2]/div[2]/div[2]/div[4]/div/label')
    hotels_toggle.click()
    for i in range(99):
        print(i)
        time.sleep(7)
        soup = BeautifulSoup(driver.page_source)
        listings = soup.find("div", id="taplc_hsx_hotel_list_lite_dusty_hotels_combined_sponsored_0").find_all("div", class_="prw_rup prw_meta_hsx_responsive_listing ui_section listItem")
        for listing in listings:
            urls.append(f'https://www.tripadvisor.com{listing.find("div", class_="listing_title").find("a")["href"]}')
            hotels.append(listing.find("div", class_="listing_title").find("a").text)
        next_button = driver.find_element_by_xpath('//*[@id="taplc_main_pagination_bar_dusty_hotels_resp_0"]/div/div/div/span[2]')
        next_button.click()
        with open("hotel_urls.pickle", "wb") as to_write:
            pickle.dump(urls, to_write)

    return urls, hotels


# For scraping reviews + metadata (no ratings) from Tripadvisor
def get_reviews(url_list):
    '''Takes a list of Tripadvisor property URLS, scrapes those property pages for data, and returns all scraped data as a df.

    Arguments: list (of URLs)
    Returns: df
    '''
    review_list_df = pd.DataFrame(columns=("Property name", "Property address", "Lat, long", "Reviewer", "Review date", "Date of stay", "Review title", "Full review", "Review link"))

    # Un-comment the below if building on top of existing df
    '''with open("alaska_hotels_redux.pickle", "rb") as to_read:
        review_list_df = pickle.load(to_read)'''

    for url in tqdm(url_list):
        driver = webdriver.Chrome(chromedriver)
        driver.get(url)
        time.sleep(6)
        
        try:
            property_lat_long = driver.find_element_by_xpath('/html/body/div[2]/div[2]/div[2]/div[5]/div/div/div/div/div/div[2]/span/img').get_attribute("src").split("center=")[1].split("&", 1)[0]
        except NoSuchElementException:
            property_lat_long = np.NaN
        soup = BeautifulSoup(driver.page_source)
        time.sleep(5)
        try:
            property_name = soup.find("h1", class_="_1mTlpMC3", id="HEADING").text
        except AttributeError:
            property_name = soup.find("h1", class_="YeV2SlB6 propertyHeading").text
        try:
            property_address = soup.find("span", class_="_3ErVArsu jke2_wbp").text
        except AttributeError:
            property_address = np.NaN
        try:
            disabled_button = soup.find("div", class_="_16gKMTFp").find(class_=re.compile(".+(disabled).*")).text
        except AttributeError:
            disabled_button = "Next"
        page_number = 1
        reviews = soup.find_all("div", class_="_2wrUUKlw _3hFEdNs8")
        if reviews == []:
            reviewer, review_date, date_of_stay, review_title, full_review, review_link = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
            review_details_list = [property_name, property_address, property_lat_long, reviewer, review_date, date_of_stay, review_title, full_review, review_link]
            index = len(review_list_df)
            review_list_df.loc[index] = review_details_list
            with open("alaska_hotels_df.pickle", "wb") as to_write:
                pickle.dump(review_list_df, to_write)
            print(f'No reviews on page {page_number} (last). Done!')
            continue
        else:
            for review in reviews:
                reviewer, review_date = review.find("div", class_="_2fxQ4TOx").find("span").text.split(" wrote a review ")
                print(reviewer)
                if review_date == "Today":
                    review_date = "November 1"
                elif review_date == "Yesterday":
                    review_date = "October 31"
                else:
                    review_date = datetime.datetime.strptime(review_date.split(" ")[0], '%b').strftime('%B') + ' ' + review_date.split(" ")[1]
                review_link = f'https://www.tripadvisor.com{review.find("div", class_="glasR4aX").a["href"]}'
                review_title = review.find("div", class_="glasR4aX").a.span.text
                full_review = review.find("q", class_="IRsGHoPm").text
                if review.find("span", class_="_355y0nZn"):
                    date_of_stay = review.find("span", class_="_34Xs-BQm").text.split(": ")[1]
                else:
                    date_of_stay = np.NaN
                review_details_list = [property_name, property_address, property_lat_long, reviewer, review_date, date_of_stay, review_title, full_review, review_link]
                index = len(review_list_df)
                review_list_df.loc[index] = review_details_list
            print(f'Page {page_number} done!')

            if disabled_button != "Next":
                try:
                    next_button = driver.find_element_by_xpath('/html/body/div[2]/div[2]/div[2]/div[7]/div/div[1]/div[1]/div/div/div[3]/div[8]/div/a')
                    next_button.click()
                except NoSuchElementException:
                    continue
                time.sleep(3.5)
                time.sleep(3)
                soup = BeautifulSoup(driver.page_source)
                while soup.find("div", id="component_14").find("div", class_="_16gKMTFp").find("div", class_="ui_pagination is-centered").find_all("a")[1].text == "Next":
                    page_number += 1
                    soup = BeautifulSoup(driver.page_source)
                    reviews = soup.find_all("div", class_="_2wrUUKlw _3hFEdNs8")
                    for review in reviews:
                        reviewer, review_date = review.find("div", class_="_2fxQ4TOx").find("span").text.split(" wrote a review ")
                        print(reviewer)
                        if review_date == "Today":
                            review_date = "November 1"
                        elif review_date == "Yesterday":
                            review_date = "October 31"
                        else:
                            review_date = datetime.datetime.strptime(review_date.split(" ")[0], '%b').strftime('%B') + ' ' + review_date.split(" ")[1]
                        review_link = f'https://www.tripadvisor.com{review.find("div", class_="glasR4aX").a["href"]}'
                        review_title = review.find("div", class_="glasR4aX").a.span.text
                        full_review = review.find("q", class_="IRsGHoPm").text
                        if review.find("span", class_="_355y0nZn"):
                            date_of_stay = review.find("span", class_="_34Xs-BQm").text.split(": ")[1]
                        else:
                            date_of_stay = np.NaN
                        review_details_list = [property_name, property_address, property_lat_long, reviewer, review_date, date_of_stay, review_title, full_review, review_link]
                        index = len(review_list_df)
                        review_list_df.loc[index] = review_details_list
                    with open("alaska_hotels_redux.pickle", "wb") as to_write:
                        pickle.dump(review_list_df, to_write)
                    print(f'Page {page_number} done!')
                    next_button = driver.find_element_by_xpath('/html/body/div[2]/div[2]/div[2]/div[7]/div/div[1]/div[1]/div/div/div[3]/div[8]/div/a[2]')
                    next_button.click()
                    time.sleep(4)
                    soup = BeautifulSoup(driver.page_source)
                
                page_number += 1
                time.sleep(3.5)
                reviews = soup.find_all("div", class_="_2wrUUKlw _3hFEdNs8")
                for review in reviews:
                    reviewer, review_date = review.find("div", class_="_2fxQ4TOx").find("span").text.split(" wrote a review ")
                    if review_date == "Today":
                        review_date = "November 1"
                    elif review_date == "Yesterday":
                        review_date = "October 31"
                    else:
                        review_date = datetime.datetime.strptime(review_date.split(" ")[0], '%b').strftime('%B') + ' ' + review_date.split(" ")[1]
                    review_link = f'https://www.tripadvisor.com{review.find("div", class_="glasR4aX").a["href"]}'
                    review_title = review.find("div", class_="glasR4aX").a.span.text
                    full_review = review.find("q", class_="IRsGHoPm").text
                    if review.find("span", class_="_355y0nZn"):
                        date_of_stay = review.find("span", class_="_34Xs-BQm").text.split(": ")[1]
                    else:
                        date_of_stay = np.NaN
                    review_details_list = [property_name, property_address, property_lat_long, reviewer, review_date, date_of_stay, review_title, full_review, review_link]
                    index = len(review_list_df)
                    review_list_df.loc[index] = review_details_list
                with open("alaska_hotels_df.pickle", "wb") as to_write:
                    pickle.dump(review_list_df, to_write)
                print(f'Page {page_number} (last) done!')

            else:
                reviews = soup.find_all("div", class_="_2wrUUKlw _3hFEdNs8")
                if reviews == []:
                    reviewer, review_date, date_of_stay, review_title, full_review, review_link = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
                    review_details_list = [property_name, property_address, property_lat_long, reviewer, review_date, date_of_stay, review_title, full_review, review_link]
                    index = len(review_list_df)
                    review_list_df.loc[index] = review_details_list
                    with open("alaska_hotels_df.pickle", "wb") as to_write:
                        pickle.dump(review_list_df, to_write)
                    print(f'No reviews on page {page_number} (last). Done!')
            driver.quit()
            time.sleep(1)

    return review_list_df


# For identifying patterns in corpus
def summarize(pattern, strings, freq):
    '''Takes a string, another string as a pattern, and a frequency, prints the 
    frequency of the pattern in the corpus.

    Arguments: string, string, int
    Returns: string
    '''
    # Find matches
    compiled_pattern = re.compile(pattern)
    matches = [s for s in strings if compiled_pattern.search(s)]
    
    # Print volume and proportion of matches
    print("{} strings which is {:.2%} of total".format(len(matches), len(matches) / len(strings)))
    
    # Create list of tuples containing matches and their frequency
    output = [(s, freq[s]) for s in set(matches)]
    output.sort(key=lambda x:x[1], reverse=True)
    
    return output


# For identifying words with extra intentional letters
def find_outlaw(word):
    '''Takes a string checks to see if it is an "oulaw" word.

    Arguments: string
    Returns: bool
    '''
    is_outlaw = False
    for i, letter in enumerate(word):
        if i > 1:
            if word[i] == word[i-1] == word[i-2] and word[i].isalpha():
                is_outlaw = True
                break
    return is_outlaw


# For spell-checking and -replacing words > 5 characters
def spell_checker(df, pickling=False):
    '''Takes a list of document strings and runs all substrings through SymSpell and replaces each
    with correctly spelled string using the dictionary (max Levenshtein distance=2).
    '''

    df = df[df["Full review"].notna()]
    
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    
    reviews = list(df["Full review"])

    cleaned_reviews = []
    for review in tqdm(reviews):
        cleaned_review = []
        for word in review.split():
            if len(word) > 5:
                word = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, \
                    transfer_casing=True, ignore_token='([A-z]+)-([A-z]+)')[0]._term
            cleaned_review.append(word)
        cleaned_reviews.append(" ".join(cleaned_review))

    df["Spell-checked review"] = cleaned_reviews
    
    if pickling==True:
    
        with open("spell_checked_data.pickle", "wb") as to_write:
            pickle.dump(df, to_write)

    return df


def review_cleaner(df, new_stopwords=None):
    '''
    Takes a df, cleans the reviews in it and returns an updated df.
    '''

    reviews = list(df["Spell-checked review"])
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(["u", "also", "mok", "eric"])
    if new_stopwords != None and type(new_stopwords) == list:
        stopwords.extend(new_stopwords)

    cleaned_reviews = []
    for review in tqdm(reviews):
        review = review.lower()
        review = re.sub('(?!\s)((\S*)((.com)|(.com\/))(\S*))', ' ', review)
        review = re.sub('[^A-Za-z\s-]', '', review)
        review = re.sub('(\W?)(\d+)(\S?)((\d+)?)(((a|p)[m])|((\s)((a|p)[m]))?)', ' ', review)
        review = re.sub('(?!\w+|\s)--+(?=\s|\w+)', ' ', review)
        review = re.sub('(?!\w+)([,]+|[.][.]+|\/+)(?=\w+)', ' ', review)
        review = re.sub('([A-Z]([a-z]+))((\s[A-Z]([a-z]+))+)', ' ', review)
        review = re.sub('-', '', review)
        review = " ".join([word for word in review.split() if word not in stopwords])
        doc = nlp(review)
        lemmatized_review = " ".join([token.lemma_ for token in doc if token.lemma_ != "-PRON-"])
        cleaned_reviews.append(lemmatized_review)

    df["Cleaned review"] = cleaned_reviews

    return df


# For additional, narrowly considered cleaning
def clean_again(review):
    '''Takes a review and removes stop words from scikit-learn's "ENGLISH STOP WORDS."
    For use with a lambda function.
    
    Arguments: string
    Returns: string
    '''

    cleaned_review = ''
    cleaned_words = 0
    for word in review.split():
        if word not in ENGLISH_STOP_WORDS:
            cleaned_review += word + ' '
        else:
            cleaned_words += 1
    print(cleaned_words)

    return cleaned_review


# For cleaning once more
def third_clean(review):
    '''Takes a review and removes custom stop words. For use with a lambda function.
    
    Arguments: string
    Returns: string
    '''

    cleaned_review = []
    stop_words = ["didst", "ovid", "thing", "place", "use", "alaska"]

    for token in review.split(" "):
        if token not in stop_words:
            cleaned_review.append(token)

    return " ".join(cleaned_review)


# For cleaning out city names from reviews
def city_clean(review):
    '''Takes a review and removes city names. For use with a lambda function.
    
    Arguments: string
    Returns: string
    '''
    
    for city in cities:
        if city in review:
            review = "".join(review.split(city))
    
    return review


# For extracting and printing the top words in a corpus
def top_words(df, num_words=10, column="Cleaned review v2", min_length=0, separater=None):
    '''
    Takes a df and an optional int (num_words) and prints the (num_words) most common words in the corpus.
    
    Arguments: df, others
    Returns: (none)
    '''
    if separater == None:
        cleaned_reviews = list(df[column])
        one_big_string = " ".join(cleaned_reviews)
        if min_length!=0:
            splits = [split for split in one_big_string.split() if len(split) > min_length]
        else:
            splits = one_big_string.split()
        freq_splits = FreqDist(splits)
        print("\n")
        for i, term in enumerate(freq_splits.most_common(num_words)):
            print(f'{i+1}. {term}')

    else:
        df = df[df[separater].notna()]
        for i, item in enumerate(df[separater].unique()):
            cleaned_reviews = list(df[df[separater] == item]["Cleaned review v2"])
            one_big_string = " ".join(cleaned_reviews)
            splits = one_big_string.split()
            freq_splits = FreqDist(splits)
            print(f'\nMost common words: {item}:')
            for i, term in enumerate(freq_splits.most_common(num_words)):
                print(f'{i+1}. {term}')


# For determining whether a review has a word or phrase in it
def get_reviews_with(review, list_of_words):
    '''Takes a string and a list of strings and returns an int (1 or 0) reflecting whether at 
    least one of the words is in the review.
    
    Arguments: string, list of strings
    Returns: int
    '''

    value = 0
    for word in list_of_words:
        if word in review:
            value = 1
            break
    return value


# For printing (via generator) and evaluating individual reviews
def aurora_review_printer():
    '''Takes no argument and returns a review via generator.

    Arguments: (none)
    Returns/yields: string

    '''

    aurora_reviews = list(nl_df["Full review"])
    length = len(aurora_reviews)
    i = 0
    while i < length:
        yield aurora_reviews[i]
        i += 1


# For creating a column of reviews lemmatized and POS-tagged as NOUN.
def get_nouns(review):
    '''Takes a string and returns only the nouns from that string. For use with 
    a lambda function..

    Arguments: df
    Returns: df
    '''
    review = nlp(review)
    review_nouns = []
    for token in review:
        if token.lemma_ != "ovid" and (token.pos_ == "NOUN" or token.pos_ == "PROPN"):
            review_nouns.append(token.lemma_)

    return " ".join(review_nouns)


# For extracting the adjectives from a review
def get_adjs(review):
    '''Takes a string and returns only the adjectives from that string. For use with 
    a lambda function.

    Arguments: string
    Returns: string
    '''

    review = nlp(review)
    review_adjs = []
    for token in review:
        if token.pos_ == "ADJ":
            review_adjs.append(token.lemma_)

    return " ".join(review_adjs)


# For getting review length
def review_length(review):
    '''Takes a review as a string and returns the review length as an int. For use in a 
    lambda function.

    Arguments: string
    Returns: int
    '''

    review_length = len(review.split())

    return review_length


# For topic modeling with NMF and LSA
def run_model(df, model, topics, ngram_range=(1,1), results_df=False):
    '''Takes a df, a model, a number of topics as an int, an n-gram range as a tuple, 
    and a bool and then runs the appropriate df column through the model and prints
    a list of topics with top included words and explained variance ratios.

    Arguments: df, model, int, tuple (optional), bool (optional)
    Returns: (none), df (optional)
    '''

    corpus = df['Cleaned review']

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.5, ngram_range=ngram_range)
    df_vectorized = tfidf.fit_transform(corpus)
    if model == "lsa":
        lsa = TruncatedSVD(topics)
        lsa_df = lsa.fit_transform(df_vectorized)

        for idx, topic in enumerate(lsa.components_):
            print(f'\nTopic {idx+1}')
            print([tfidf.get_feature_names()[i] for i in (topic.argsort()[-5::1])])
            print(f'Explained variance ratio: {lsa.explained_variance_ratio_[idx]}')

    elif model == "nmf":
        nmf = NMF(topics)
        doc_topic = nmf.fit_transform(df_vectorized)

        for idx, topic in enumerate(nmf.components_):
            print(f'\nTopic {idx+1}')
            print([tfidf.get_feature_names()[i] for i in (topic.argsort()[-5::1])])
        
        if results_df==True:
            topic_df = pd.DataFrame(doc_topic)
            topic_df[["Full review", "Cleaned review"]] = df[["Full review", "Cleaned review"]]

            return topic_df


# For getting Vader sentiment for each review
def vader_scores(df):
    '''Takes a df with a 'Cleaned review' column, runs Vader on the review
    and returns polarity scores for each review in new columns'''

    reviews = list(df["Cleaned review"])
    vader = SentimentIntensityAnalyzer()
    vader_pos, vader_neg = [], []
    for review in tqdm(reviews):
        score = vader.polarity_scores(review)
        vader_pos.append(score['pos'])
        vader_neg.append(score['neg'])

    df["Vader +"] = vader_pos
    df["Vader -"] = vader_neg

    return df


# For getting seasons adjectives
def get_seasonal_adjs(dictionary):
    '''Takes a dict and assigns a Vader score to each word (if not already in the defaultdict 
    returns a df with a season column assigned to each review.

    Arguments: dict
    Returns: df
    '''

    vader = SentimentIntensityAnalyzer()

    seasonal_adj_df = pd.DataFrame(columns=["Month", "Season", "Corpus"])
    vader_score_dict = defaultdict()
    for i in tqdm(range(1, 13)):
        month_adjs = adjs_by_month[f'Month {i}']
        month_corpus = []
        for item in month_adjs:
            word = item.split("'")[1]
            appearances = int(item.split("', ")[1].strip(")"))
            for _ in range(appearances):
                month_corpus.append(word)
            score = vader.polarity_scores(word)

            if word not in vader_score_dict:
                vader_score_dict[word] = (score['pos'], score['neg'], (score['pos']-score['neg']))
        if i in [12, 1, 2]:
            season = "Winter"
        elif i in [3, 4, 5]:
            season = "Spring"
        elif i in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"
        
        month_details = {"Month": i, "Season": season, "Corpus": " ".join(month_corpus)}
        seasonal_adj_df = seasonal_adj_df.append(month_details, ignore_index=True)

    return seasonal_adj_df


# For getting season of an individual review
def get_season(i):
    '''Takes an int or float (i) representing a "Month of Stay" and returns an int 
    representing the "Season of Stay."

    Arguments: int/float
    Returns: int
    '''

    if i in [12, 1, 2]:
        season = "Winter"
    elif i in [3, 4, 5]:
        season = "Spring"
    elif i in [6, 7, 8]:
        season = "Summer"
    elif i in [9, 10, 11]:
        season = "Fall"
    else:
        season = np.NaN

    return season


# For creating a class for use in building a wordcloud in the shape of Alaska and colored according
# to pos/neg categorization by Vader 
class GroupedColorFunc(object):
    '''Create a color function object which assigns different shades of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.

    # BORROWED
    '''

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        '''Returns a single_color_func associated with the word
        '''

        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


# For actually printing the above wordcloud
def get_seasonal_wordcloud(df, season, column="Clean review no cities", pos_color='#92F0B2', neg_color='#FD99B2', def_color='#BDE4FD'):
    '''Takes a df, a chosen season (see rest of doc) and other optional parameters and returns a wordcloud in the shape of Alaska (for now)
    with colors correlating with Vader scores.
    '''

    seasonal_df = pd.DataFrame(columns=["Month", "Season", "Corpus"])
    vader_score_dict = defaultdict()
    for i in tqdm(range(1, 13)):
        review_list = list(df[df["Month of stay"] == i][column])
        month_list = []
        for group in review_list:
            for adj in group.split(" "):
                month_list.append(adj)
        for review in list(set(month_list)):
            for word in review.split(" "):
                score = vader.polarity_scores(word)
                if word not in vader_score_dict:
                    vader_score_dict[word] = (score['pos'], score['neg'], (score['pos']-score['neg']))
        if i in [12, 1, 2]:
            mon_season = "Winter"
        elif i in [3, 4, 5]:
            mon_season = "Spring"
        elif i in [6, 7, 8]:
            mon_season = "Summer"
        else:
            mon_season = "Fall"
        
        month_details = {"Month": i, "Season": mon_season, "Corpus": " ".join(month_list)}
        seasonal_df = seasonal_df.append(month_details, ignore_index=True)

    corpus = " ".join(list(seasonal_df[seasonal_df["Season"] == season]["Corpus"]))

    positive_words = [word for word in corpus.split(" ") if word in vader_score_dict and vader_score_dict[word][2] == 1]
    negative_words = [word for word in corpus.split(" ") if word in vader_score_dict and vader_score_dict[word][2] == -1]
    print("Moving to wordcloud generation")
    text = corpus
    AK_shape = np.array(Image.open('Alaska shape.png'))
    wc = WordCloud(mask=AK_shape, background_color="black",
                max_words=500, max_font_size=75,
                random_state=10, width=AK_shape.shape[1],
                height=AK_shape.shape[0]).generate(text)

    color_to_words = {pos_color: positive_words, neg_color: negative_words}
    default_color = def_color
    grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

    wc.recolor(color_func=grouped_color_func)
    plt.figure(figsize=(20, 20))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show();

    return vader_score_dict

        

# I built this early on and sort of abandoned it but I'm leaving here in case I return to it
class AuroraPipeline:
    

    def __init__(self, df, vectorizer=CountVectorizer(min_df=5), cleaning_function=None, model=None):
        '''
        A class for pipelining all relevant review data for cleaning/pre-processing, vectorizing,
        training, fitting, and transforming.

        Arguments:
        '''
        self.reviews = list(df["Full review"])

        if not tokenizer:
            tokenizer = self.splitter
        self.tokenizer = tokenizer
        if not cleaning_function:
            cleaning_function = self.clean_text
        
        self.model = model
        self.cleaning_function = cleaning_function
        self.vectorizer = vectorizer
        self._is_fit = False


    def aurora_only(self):
        '''
        Replaces the operating data with reviews that mention "northern lights" or "aurora."
        '''
        self.reviews = list(df[(df["Full review"].str.lower().str.contains("northern lights")) | (df["Full review"].str.lower().str.contains("aurora"))])


    def review_cleaner(self, df):
        '''
        Takes the reviews in question, cleans them and returns a new list of reviews.
        '''

        reviews = self.reviews

        cleaned_reviews = []
        for review in tqdm(reviews):
            review = review.lower()
            review = re.sub('(?!\s)((\S*)((.com)|(.com\/))(\S*))', ' ', review)
            review = re.sub('[^A-Za-z\s-]', '', review)
            review = re.sub('(\W?)(\d+)(\S?)((\d+)?)(((a|p)[m])|((\s)((a|p)[m]))?)', ' ', review)
            review = re.sub('(?!\w+|\s)--+(?=\s|\w+)', ' ', review)
            review = re.sub('(?!\w+)([,]+|[.][.]+|\/+)(?=\w+)', ' ', review)
            review = re.sub('([A-Z]([a-z]+))((\s[A-Z]([a-z]+))+)', ' ', review)
            review = re.sub('-', '', review)
            review = [word for word in review.split() if word not in stopwords]
            review = " ".join([wnl.lemmatize(word) for word in review])
            cleaned_reviews.append(review)

        self.reviews = cleaned_reviews


    def top_words(self, num_words=10):
        '''
        Takes an int (num_words) and prints the (num_words) most common words in the corpus.
        '''
        cleaned_reviews = self.reviews

        one_big_string = " ".join(cleaned_reviews)
        splits = one_big_string.split()
        freq_splits = FreqDist(splits)
        print("\n")
        for i, term in enumerate(freq_splits.most_common(num_words)):
            print(f'{i+1}. {term}')

    
    def spell_checker(self):
        '''Takes a list of document strings and runs all substrings through SymSpell and replaces each
        with correctly spelled string using the dictionary (max Levenshtein distance=2).
        '''

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
        
        reviews = self.reviews

        cleaned_reviews = []
        for review in tqdm(list_of_reviews):
            cleaned_review = []
            for word in tqdm(review.split()):
                if len(word) > 5:
                    word = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, \
                        transfer_casing=True, ignore_token='([A-z]+)-([A-z]+)')[0]._term
                cleaned_review.append(word)
            cleaned_reviews.append(" ".join(cleaned_review))

        self.reviews = cleaned_reviews

    def run_lsa(self, topics):
        
        tfidf = TfidfVectorizer(stop_words="english", max_df=0.5)
        df_vectorized = tfidf.fit_transform(self.reviews)
        lsa = TruncatedSVD(topics)
        lsa_df = lsa.fit_transform(df_vectorized)

        for idx, topic in enumerate(lsa.components_):
            print(f'Topic {idx+1}\n')
            print([tfidf.get_feature_names()[i] for i in (topic.argsort()[-(topics)::1])])



