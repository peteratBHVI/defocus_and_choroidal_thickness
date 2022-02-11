# defocus_and_choroidal_thickness
Code documentation of study: effects of optical defocus on choroidal thickness

Purpose: to compile and to analyses topographical choroidal maps 

Jupyter notebook, Python:   average_rnfl_layers_after_im_registration.jpynb

Aim: find macula centre (low RNFL thickness) for landmark matching of choroidal thickness maps

Jupyter notebook, Python:   gif_from_projection_of_anterior_segment.jpynb

Aim: create gif from intra participant z-projection of anterior segment of the eye from OCT scans to estimate quality of transformation, i.e., finding shifts between scans

Jupyter notebook, Python:   averaged_choroid_thickess_data_from_reg_scansV2.jpynb

Aim: use transformation info from z-projections to align intra participant choroidal thickness maps

Jupyter notebook, Python:   macula_matching_averaging_of_layer_thickness_V3.jpynb

Aim: compile topographical choroidal thickness map
-	Use macula centre position to match inter participant choroidal thickness maps 
-	Disregard data from optic nerve head using annotated z-projections
-	Apply outlier filtering
-	Calculating residuals for post and pre intervention

Jupyter notebook, Python:   choroidal_thickness_residual_visualisation.jpynb

Aim: visualise data and proceed stats

-	Statistical analyses on interparticipant landmark matched choroid thickness residual data
-	Visualization of the residual of interparticipant landmark matched choroid thickness data post and pre intervention

Dependencies of jpynb:

from oct_helpers_lib import OctDataAccess as get_px_meta
-	Easy access of participant’s meta data e.g., file location 
from oct_helpers_lib import TopconSegmentationData as TSD
-	Extracting specific data from OCT data extractor output file
