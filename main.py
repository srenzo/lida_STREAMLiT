from genericpath import exists
from queue import Empty
import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd
from googletrans import Translator

translator = Translator()

options = ["PT", "EN"]

if 'prompt' not in st.session_state:
    pressed_buttons = []
    st.session_state['prompt'] = pressed_buttons  # Default language

if 'language' not in st.session_state:
    st.session_state['language'] = 'PT'  # Default language
    st.session_state['idx'] = 0

##print("\n1.>", st.session_state.language, " | ", str(st.session_state.idx))

def t2(text):
    translation = translator.translate(text, dest=st.session_state.language.lower())
    return str(translation.text)

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="LIDA", page_icon="📊",
)

### select language
##st.sidebar.write(t2("## Language"))
##langs = ["pt", "en", "es"]
##target_lang = st.sidebar.selectbox(
##    t2('Choose a language'),
##    options=langs,
##    index=0
##)

# Buttons on the sidebar for language selection
with st.sidebar:
    idx = st.session_state.idx
    lang = st.radio(
        t2("Select language:"),
        options,
        index=idx,
        label_visibility="collapsed",
        horizontal=True
    )
    st.session_state['language'] = lang
    idx = options.index(lang)

##print("\n2.>", lang, ' | ', st.session_state.language, " | ", str(idx))

st.write(t2("# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models") + " 📊")
st.sidebar.write(t2("## Setup"))

# Step 1 - Get OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    if openai_key:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        st.sidebar.write(t2(f"Current key: {display_key}"))
    else:
        st.sidebar.write(t2("Please enter OpenAI API key."))
else:
    display_key = openai_key[:2] + "***" + openai_key[-3:]
    st.sidebar.write(t2(f"OpenAI API key loaded from environment variable: {display_key}"))

st.markdown(t2(
    """
    LIDA is a library for generating data visualizations and data-faithful infographics.  
    LIDA is grammar agnostic (will work with any programming language and visualization  
    libraries e.g., matplotlib, seaborn, altair, d3, etc) and works with multiple LLM providers (OpenAI, Azure OpenAI, PaLM, Cohere, Huggingface).    
    Details on the components of LIDA are described in the (https://arxiv.org/abs/2303.02927).  
    See the project page (https://microsoft.github.io/lida/) for updates !
   
"""))

# Step 2 - Select a dataset and summarization method
if openai_key:
    # Initialize selected_dataset to None
    selected_dataset = None

    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write(t2("## Text Generation Model"))
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    if 'model' not in st.session_state:
        idx = 2
    else:
        idx = st.session_state['model']
    selected_model = st.sidebar.selectbox(
        t2('Choose a model'),
        options=models,
        index=idx
    )
    st.session_state['model'] = models.index(selected_model)
    ###print("\n--->", st.session_state['model'])

    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    if 'temp' not in st.session_state:
        temp = 0.5
    else:
        temp = st.session_state['temp']
    temperature = st.sidebar.slider(
        t2("Temperature"),
        min_value=0.0,
        max_value=1.0,
        value=temp)
    st.session_state['temp'] = temperature
    ###print("\n--->", st.session_state['temp'])

    # set use_cache in sidebar
    if 'cache' not in st.session_state:
        cache = True
    else:
        cache = st.session_state['cache']
    use_cache = st.sidebar.checkbox(t2("Cache Enabled"), value=cache)
    st.session_state['cache'] = use_cache

    # Handle dataset selection and upload
    st.sidebar.write(t2("## Data Summarization"))
    st.sidebar.write(t2("### Choose a dataset"))

    if 'datasets' not in st.session_state:
        datasets = [
        {"label": t2("Select a dataset"), "url": None},
        {"label": t2("Cars"), "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": t2("Weather"), "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]
    else:
        datasets = st.session_state['datasets']
        selected_dataset = st.session_state['dataset']

    selected_dataset_label = st.sidebar.selectbox(
        t2('Choose a dataset'),
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox(t2("Upload your own data"))

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader(t2("Choose a CSV or JSON file"), type=["csv", "json"])

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            st.session_state['datasets'] = datasets

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info(t2("To continue, select a dataset from the sidebar on the left or upload your own."))

    st.session_state['dataset'] = selected_dataset

    st.sidebar.write(t2("### Choose a summarization method"))
    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {"label": "llm",
         "description":
         t2("Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description")},
        {"label": "default",
         "description": t2("Uses dataset column statistics and column names as the summary")},

        {"label": "columns", "description": t2("Uses the dataset column names as the summary")}]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        t2('Choose a method'),
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {t2(selected_summary_method_description)} </span>",
            unsafe_allow_html=True)

# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method:
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

    st.write(t2("## Summary"))
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    ###print("\nsummary: ", summary)

    if t2("dataset_description") in summary:
        st.write(t2(summary["dataset_description"]))

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    # Step 4 - Generate goals
    if summary:
        st.sidebar.write(t2("### Goal Selection"))

        num_goals = st.sidebar.slider(
            t2("Number of goals to generate"),
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox(t2("Add Your Own Goal"))

        # **** lida.goals *****
        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
        st.write(t2("## Goals ") + f"({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [t2(goal.question) for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input(t2("Describe Your Goal"))

            if user_goal:

                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox(t2('Choose a generated goal'), options=goal_questions, index=0)

        # st.markdown("### Selected Goal")
        selected_goal_index = goal_questions.index(selected_goal)
        ###print("\n----------------------->",selected_goal_index, " | ", goals[selected_goal_index])
        st.write(t2(goals[selected_goal_index].rationale))

        selected_goal_object = goals[selected_goal_index]

        # Step 5 - Generate visualizations
        if selected_goal_object:
            st.sidebar.write(t2("## Visualization Library"))
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = st.sidebar.selectbox(
                t2('Choose a visualization library'),
                options=visualization_libraries,
                index=0
            )

            # Update the visualization generation call to use the selected library.
            st.write(t2("## Visualizations"))

            # slider for number of visualizations
            num_visualizations = st.sidebar.slider(
                t2("Number of visualizations to generate"),
                min_value=1,
                max_value=10,
                value=2)

            textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # **** lida.visualize *****
            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            ###print("\n===========================>>", visualizations)

            viz_titles = [t2('Visualization') + f'{i+1}' for i in range(len(visualizations))]

            selected_viz_title = st.selectbox(t2('Choose a visualization'), options=viz_titles, index=0)

            selected_viz = visualizations[viz_titles.index(selected_viz_title)]
            ###print("\n===============>", selected_viz)

            # image placeholder
            imageLocation = st.empty()

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                ##st.image(img, caption=selected_viz_title, use_column_width=True)
                imageLocation.image(img, caption=selected_viz_title, use_column_width=True)


        # Step 6 - Modify visualizations using LLM
        if selected_viz.raster:
            # Modify options
            st.write(t2("## Suggestions of LLM 'PROMPT' to modify visualizations"))
            ####prompt = st.text_input(t2('LLM PROMPT'), '')
            # modify chart using natural language
            ###instructions = ["convert this to a bar chart", "change the color to red", "change y axes label to Fuel Efficiency", "translate the title to french"]
            prompt = st.empty()
            max_idx = 5
            suggestions=[t2('convert this to a pie chart'), t2('change the color to pastels tones'), t2('translate the title to brazilian portuguese')]
            pressed_buttons = st.session_state.prompt
            for suggestion in suggestions:
                if st.button(suggestion):
                    # Update the text input with the suggestion when its button is clicked
                    pressed_buttons.append(suggestion)
                    st.session_state['prompt'] = pressed_buttons
            prompt = prompt.text_input(t2('LLM PROMPT'), pressed_buttons)
            ###prompt = pressed_buttons
            ###print("\n----->>", prompt)
            edited_chart = lida.edit(code=selected_viz.code, summary=summary, instructions=prompt, library=selected_library, textgen_config=textgen_config)
            ###print("\n===============>", edited_chart[0])
            imgdata = base64.b64decode(edited_chart[0].raster)
            img = Image.open(io.BytesIO(imgdata))
            ##st.image(img, caption=selected_viz_title, use_column_width=True)
            imageLocation.image(img, caption=selected_viz_title, use_column_width=True)

            ### generate explanation for chart
            ###st.write(t2("## Visualization Explanation"))
            ###explanation = lida.explain(code=selected_viz.code, summary=summary)

            # Infographic Generation [WIP]
            infographics = lida.infographics(visualization = selected_viz.raster, n=3, style_prompt="line art")

            st.write(t2("### Code for Visualization"))
            st.code(selected_viz.code)
