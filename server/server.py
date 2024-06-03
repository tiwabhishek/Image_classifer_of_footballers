from flask import Flask, request, jsonify  # Import necessary modules from Flask
import util  # Import the util module which contains our custom functions

# Initialize a new Flask web application
app = Flask(__name__)

# Define a route for the endpoint '/classify_image' which accepts both GET and POST requests
@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    # Extract image data from the form data in the request
    image_data = request.form['image_data']
    
    # Call the classify_image function from the util module to get the classification results
    classification_result = util.classify_image(image_data)
    
    # Create a JSON response with the classification results
    response = jsonify(classification_result)
    
    # Add a header to the response to allow cross-origin requests
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    # Return the JSON response
    return response

# Entry point of the script
if __name__ == "__main__":
    # Print a message indicating that the Flask server is starting
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    
    # Load any saved artifacts or models using a function from the util module
    util.load_saved_artifacts()
    
    # Run the Flask web server on port 5000
    app.run(port=5000)
