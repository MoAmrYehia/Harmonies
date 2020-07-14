## Segmenting persons out of an image
### Installation
Required packages:
- Flask
- Matplotlib
- PIL
- torch >= 1.5
- detectron2 [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html)
- Numpy

### Running
```
$ python app.py 
```

### Request
Then send the following JSON request on `localhost:5000` 

```json
{
        "img": "adfklk..."
}
```

### Response
The response text will be the base64 representation of the masked PNG image
