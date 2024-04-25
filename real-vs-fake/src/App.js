import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImage(file);
  };

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    if (!image) {
      setError('Please select an image.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data.result);
      setLoading(false);
    } catch (error) {
      setError('Error uploading image. Please try again.');
      setLoading(false);
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>Deepfake Detection App</h1>
      </header>
      <main>
        <form onSubmit={handleFormSubmit}>
          <div className="upload-container">
            <div className="image-preview">
              {image ? (
                <img src={URL.createObjectURL(image)} alt="Selected" />
              ) : (
                <div className="empty-box"></div>
              )}
            </div>
            <label htmlFor="file-upload" className="file-input-label">Choose a file</label>
            <input
              type="file"
              id="file-upload"
              accept="image/*"
              onChange={handleImageUpload}
              className="file-input"
            />
            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Loading...' : 'Submit'}
            </button>
          </div>
        </form>
        {error && <div className="error">{error}</div>}
        {result && (
          <div className="result-container">
            <h2>Result:</h2>
            <p className={`result ${result === 'Fake' ? 'fake' : 'real'}`}>{result}</p>
          </div>
        )}
      </main>
      <footer>
      <p>AISSMS IOIT</p>
        
      </footer>
    </div>
  );
}

export default App;
