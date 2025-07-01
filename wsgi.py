import os
from app import app  # Importing the Flask app object

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))  # Get port from environment or default to 4000
    app.run(host="0.0.0.0", port=port)        # Run the app on all IPs (for deployment)
