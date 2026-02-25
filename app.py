# Imports + pipeline loader

import streamlit as st
import rag_pipeline

# Page config + session state

st.set_page_config(
    page_title="Clinical Lab Assistant",
    page_icon="🔬",
    layout="centered"
)

if "history" not in st.session_state:
    st.session_state.history = []

# Page header + input area

st.title("Clinical Lab Assistant")
st.markdown("Ask questions about common laboratory tests — preparation, results, reference ranges, and more.")

with st.form("query_form"):
    query = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Submit")

# Query handling

if submitted and query:
    with st.spinner("Searching..."):
        response = rag_pipeline.route_query(query)
    if not isinstance(response, str):
        answer = st.write_stream(response.response_gen)
    else:
        answer = response
        st.write(answer)
    if not isinstance(response, str):
        with st.expander("View retrieved sources"):
            for node in response.source_nodes:
                st.markdown(f"**Source:** {node.metadata['source']} | **Test:** {node.metadata['test_name']}")
                st.caption(node.text)
    st.session_state.history.append({"question": query, "answer": answer})

# History display

if st.session_state.history:
    st.subheader("Previous Questions")
    # Display newest query at the top while excluding current query
    # Guards against showing the current answer twice
    for item in reversed(st.session_state.history[:-1]):
        st.markdown(f"**Q:** {item['question']}")
        st.write(item['answer'])
        st.divider()


