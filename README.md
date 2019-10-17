# Simple image segmentation with k-Means

<table>
<tr>
<th>Sample input</th>
<th>Output</th>
</tr>
<tr>
<td>
<img src="data/test.jpg"/>
</td>
<td>
<img src="data/test_segmentation.png"/>
</td>
</tr>
</table>

## Setup

```bash
git clone git@github.com:sdll/simple-image-segmentation.git
cd simple-image-segmentation
virtualenv --no-site-packages venv
. venv/bin/activate
pip install opencv-python numpy matplotlib sklearn
```

## Segmenting an image

```bash
python segment.py --image data/test.jpg
```

## References

For the context behind EM for k-Means clusterization, see [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html) by [@jakevdp](https://twitter.com/jakevdp).