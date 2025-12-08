from app import *

def about_page():
    """Renders the content for the Blog Page."""
    st.title("💡 Blog Page")
    st.write("Welcome to the blog page! Here you can read the latest blog posts.")
    st.markdown("---")
    st.markdown("### Placeholder for your Blog Page")
    st.info("You would add your blog posts here.")
    if st.button("← Back to Home"):
        navigate_to('home')