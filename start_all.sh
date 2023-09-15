# Commands and logic of the script
echo "Hello, I am start_all.sh"

# Run script1.sh
# source start_download.sh

# Or using the dot operator
. start_download.sh

# Continue with more commands after running script1.sh
echo "start_download.sh has been executed."

. start_train.sh

echo "start_train.sh has been executed."

echo "start_all.sh has been executed."
