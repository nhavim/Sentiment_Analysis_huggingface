import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Functions
def main():
    st.title("Covid Tweets Sentiment Analysis NLP App")
    st.subheader("Team Harmony Project")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key="nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                tokenizer = AutoTokenizer.from_pretrained("Abubakari/finetuned-Sentiment-classfication-ROBERTA-model")
                model = AutoModelForSequenceClassification.from_pretrained("Abubakari/finetuned-Sentiment-classfication-ROBERTA-model")

                # Tokenize the input text
                inputs = tokenizer(raw_text, return_tensors="pt")

                # Make a forward pass through the model
                outputs = model(**inputs)

                # Get the predicted class and associated score
                predicted_class = outputs.logits.argmax().item()
                score = outputs.logits.softmax(dim=1)[0][predicted_class].item()

                # Compute the confidence level
                # Compute the confidence level

                confidence_level = np.max(outputs.logits.detach().numpy())


                # Print the predicted class and associated score
                st.write(f"Predicted class: {predicted_class}, Score: {score:.3f}, Confidence Level: {confidence_level:.2f}")

                # Emoji
                if predicted_class == 2:
                    st.markdown("Sentiment: Positive :smiley:")
                elif predicted_class == 1:
                    st.markdown("Sentiment: Neutral :😐:")
                else:
                    st.markdown("Sentiment: Negative :angry:")

            # Create the results DataFrame
            results_df = pd.DataFrame({
                "Sentiment Class": [predicted_class],
                "Score": [score]
            })

            # Create the Altair chart
            chart = alt.Chart(results_df).mark_bar().encode(
                x="Sentiment Class",
                y="Score"
            )

            # Display the chart
            with col2:
                st.altair_chart(chart, use_container_width=True)

    else:
        st.subheader("About")
        st.write("This is a sentiment analysis NLP app developed by Team Harmony for analyzing tweets related to Covid-19. It uses a pre-trained RoBERTa model to predict the sentiment of the input text. The app is part of a project to promote teamwork and collaboration among developers.")

if __name__ == "__main__":
    main()
