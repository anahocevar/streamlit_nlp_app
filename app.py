import streamlit as st
from joblib import load
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

def make_wordcloud_plot(top_words):
    
    cloud_language = WordCloud(random_state=10)
    cloud_language.generate_from_frequencies(top_words[0])
    cloud_animal = WordCloud(random_state=10)
    cloud_animal.generate_from_frequencies(top_words[1])
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cloud_language)
    ax[0].set_title('python the language')
    ax[1].imshow(cloud_animal)
    ax[1].set_title('python the animal')
    plt.grid(False)
    
    return fig
    

def app():
    
    st.title("Simple streamlit demo")
    
    st.markdown(""" ### This is a simple app with which you can:
    * get a prediction on whether your short text refers to python the programming language or python the animal,
    * learn more about the model and what it is picking up on to make predictions 
    """)
    
    st.sidebar.markdown('''#### Tools used in this project:''')
    st.sidebar.markdown('''
    - [PRAW](https://praw.readthedocs.io/en/stable/)
    - [scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
    - [Streamlit](https://streamlit.io/)
    - [GitHub repo of this project](https://github.com/anahocevar/streamlit_nlp_app)''')
    image = Image.open('images/pic.png')
    st.image(image)


    
    text = st.text_input('Write your input in box below and press enter to get a prediction:', '')
    
    if text: 
        pipe = load('models/classifier.joblib')
        prediction  = pipe.predict([text])
        st.write(f'The model is predicting that the text refers to python the **{prediction[0]}**.')
    
        with open('models/top_words.pkl', 'rb') as f:
            top_words = pickle.load(f)
        
        figure = make_wordcloud_plot(top_words)
        st.write('The model is a simple bag-of-words model together with a naive Bayes. It learns what words most disproportionately appear in one class of documents compared to the other class. We calculated the log of the ratio of the conditional probabilities for each word and below we are showing a word could of the results.')
        st.pyplot(figure)
        
        st.write('Training data was obtained from Reddit, namely subreddits "python" and "ballpython" which refer to the programming language and the animal, respectively.')
        st.write('\n')
        st.write('The model achieved around 92% accuracy on the test set.')
    
    
if __name__ == '__main__':
    app()