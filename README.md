## simglucose

This repository contains a modified version of the original **simglucose** simulator:  
https://github.com/jxx123/simglucose

### Modifications

The following changes have been introduced:

1.  **Higher interaction frequency** with the simulator, enabling finer control and more detailed experimentation.

Example: [`examples/different_interaction_frequency.py`](https://github.com/Cho-Geonwoo/simglucose/blob/master/examples/different_interaction_frequency.py)

2.  **Modified observation definition**  
    Instead of returning the _average glucose level_, the simulator now returns the **most recent blood glucose measurement**, which preserves the original signal without distortion.

3.  **Optional observation noise**  
    Users can choose whether to add noise to the glucose observations.

Example: [`examples/no_noise_in_observation.py`](https://github.com/Cho-Geonwoo/simglucose/blob/master/examples/no_noise_in_observation.py)
