import './App.css';
import React, { useState } from "react";


const ShowImage = (props) => {
  if (props.image) {
    return (
      <img style={props.style} src={props.image.preview}/>
    );
  }
};

function App() {
  const [file, setFile] = useState(null);
  const [painting, setPainting] = useState(null);

  const onFileChange = event => {
    if (event.target.files[0]) {
      const img = {
        preview: URL.createObjectURL(event.target.files[0]),
        data: event.target.files[0],
      }
      setFile(img);
    }
    else {
      setFile(null);
    }
  }

  const onFileUpload = async (e) => {
    e.preventDefault();
    if (file != null) {
      const formData = new FormData();
      console.log(file.data);
      formData.append('file', file.data);
      const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      const b64string = 'data:image/jpeg;base64,' + data['painting'].split('\'')[1];

      setPainting(b64string);
    }
    else {
      alert("No file submitted");
    }
  }

  return (
    <div className="App">
      <h1>Monet Painting Generator Using CycleGAN!</h1>
      <h3>Submit your image:</h3>
      <div>
        <input type="file" onChange={onFileChange} />
        <button onClick={onFileUpload}>
          Generate!
        </button>
      </div>
      <ShowImage style={{ maxWidth: '40%' }} image={file}/>
      <img style={{ maxWidth: '40%' }} src={painting}/>
    </div>
  );
}

export default App;
