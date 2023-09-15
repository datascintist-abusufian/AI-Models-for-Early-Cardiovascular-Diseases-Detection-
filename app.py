import streamlit as st
from  predict_page import main
from explore_page import show_explore_page


page = st.sidebar.selectbox("Explore or predict", ("Predict", "Explore"))


if page == "Predict":
    main()
else:
    show_explore_page()