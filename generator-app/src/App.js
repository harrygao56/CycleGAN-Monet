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

  const paneStyles = {
    width: '90vw',
    borderRadius: '20px',
    margin: 'auto',
    marginTop: '35px',
    background: 'rgb(183,255,252)',
    padding: '25px',
    boxShadow: '0px 3px 10px',

  }

  return (
    <div className="App">
      <div style={paneStyles}>
        <h1>Monet Painting Generator Using CycleGAN!</h1>
        <p>Created by Harry Gao</p>
        <h3>Submit your image:</h3>
        <div>
          <input type="file" onChange={onFileChange} />
          <button onClick={onFileUpload}>
            Generate!
          </button>
        </div>
      </div>
      <div style={{marginTop: '20px'}}>
        <ShowImage style={{ maxWidth: '40%', margin: '25px', boxShadow: '0px 5px 10px'}} image={file}/>
        <img style={{ maxWidth: '40%', margin: '25px', boxShadow: '0px 5px 10px'}} src={painting}/>
      </div>
    </div>
  );
}

export default App;
