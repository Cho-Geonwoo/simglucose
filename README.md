## simglucose

This repository contains a modified version of the original **simglucose** simulator:  
https://github.com/jxx123/simglucose

### Modifications

The following changes have been introduced:

1. **Higher interaction frequency** with the simulator, enabling finer control and more detailed experimentation.

2. **Modified observation definition**  
   Instead of returning the *average glucose level*, the simulator now returns the **most recent blood glucose measurement**, which preserves the original signal without distortion.

3. **Optional observation noise**  
   Users can choose whether to add noise to the glucose observations.