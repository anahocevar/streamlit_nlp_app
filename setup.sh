mkdir -p ~/.streamlit/

# You can replace the email address below with your own.
# It won't affect the app's functionality at all, but you may as well.
echo "\
[general]\n\
email = \"ahocevara@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml