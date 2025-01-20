> Image Adaptation to aid People with Visual Difficulties
>
> Sai Venkat Reddy Sheri˚, Surya Suhas Reddy Sathi:, Chaitanya Sai
> Chandu Yendru: University at Buffalo, Buffalo, NY, USA
>
> ssheri@buffalo.edu suryasuh@buffalo.edu cyendru@buffalo.edu

Abstract—Millions of global citizens suffer from myopia and
hypermetropia, hindering daily activities involving digital inter-faces.
These websites help cater to different device displays, but they do not
support vision-related issues and necessitate users to wear glasses
externally. This project is set out to devise a software-based solution
that aims to improve digital accessibility directly for the visually
impaired from websites. As it implements advanced algorithms in computer
vision and image processing, for instance, warping, contrast
enhancement, edge detection, depth-of-field adjustments, and contextual
super-resolution, the system computationally re-renders on-screen
visuals for individ-ual vision profiles. Unlike traditional physical
aids, this method allows for personal tuning of the digital viewing
experience under some conditions, such as myopia degree/hypermetropia.
With modern technological capabilities, the website implementing the
algorithms developed enables users to experience corrected visu-als,
simulated colour blindness, and automated image captioning, thus
improved digital engagement despite difficulties in vision. This
user-centric, inclusive digital solution could revolutionize
accessibility by dynamically enhancing website visuals as per users’
needs—a capability currently overlooked by web develop-ers solely
focused on device compatibility.

Index Terms—Visual impairments, Myopia, Hypermetropia, Digital
accessibility, Computer vision, Image processing tech-niques, Automated
image captioning, Super-resolution, Correc-tive lens simulation

> I. INTRODUCTION

In the current digital era, there are many types of visual impairments:
myopia and hypermetropia. These defects are a big challenge to millions
of people worldwide. These defects are a serious hindrance to performing
the regular digital activities, such as reading text on a web page,
facial recognition in images, or discriminating significant visual cues
in online content. Websites have already adapted their designs and
layouts to be responsive for different devices and screen sizes.
However, most of them do not consider the users with vision defects who
are supposed to overcome these problems by using glasses or contact
lenses.

This project tries to fill the above gap by creating a software solution
to improve the digital accessibility of vision-impaired users directly
through the websites that they use. It achieves this through the use of
advanced computer vision and image processing techniques—image warping,
contrast enhancement, edge detection, depth-of-field adjustments, and
contextual super-resolution—to re-render on-screen visuals in real-time,
optimizing it according to each user’s specific vision profile.

Unlike traditional physical solutions, which provide a one-size-fits-all
solution, this innovative solution allows customiza-

tion of the digital viewing experience according to individual users’
specific condition and degree of visual impairment. By leveraging the
capabilities of modern technological functions, a website that employs
the algorithms developed would allow users to experience corrected
visuals, simulated color blind-ness, and automated image captioning,
which facilitate better digital participation despite their vision
difficulties.

The main aim of the current project is to develop a user-centered,
inclusive digital solution. This solution changes the dynamics of
accessibility by rendering the visual of web pages dynamically as per
the specific needs of each user. Currently, most websites aim for device
compatibility rather than accommodating the vast variety of visual
abilities among users.

By solving this critical problem, the proposed system is expected to
empower visually impaired individuals so that they may experience the
digital world without external aids. This project can be regarded as a
milestone toward creating an inclusive and accessible digital online
environment in which all may equitably access digital content.

> II\. LITERATURE SURVEY

There are some important studies done on in this sector, the few
important studies include Personalising web page presentation for older
people (2006), in this paper they have discussed about the structure of
the websites, what are the standard structures and what are the changes
that we can make to help the elder people.

In this paper they mentions the use of W3C document for structured
representations of loaded HTML documents , reducing development load on
mediator but this will require installation on local machine.

The next paper User Friendly Websites In The Eyes Of Young And Old
People which is published in 2009 focus on whether elderly individuals
navigate websites differently from younger people, focusing on usability
studies, particularly eye-tracking studies, related to older users. It
discusses the analytical tool offered by Kress Van Leeuwen for analyzing
images and how it can be applied to understand differences in website
navigation between younger and older users. The authors propose
utilizing data from eye-tracking studies or observational studies of
navigational behavior to gain insights into reading patterns, time spent
on tasks, and the impact of website elements on user access to
information.

The paper does not address potential ethical considerations or privacy
concerns related to conducting eye-tracking studies

<img src="./hixg01ab.png" style="width:2in;height:1.328in" />

or observational research on website users,There is a lack of discussion
on the generalizability of the findings to broader populations beyond
the specific age groups studied.

One notable paper in the field of websites for Color blind people is ”A
Study of Color Transformation on Website Images for the Color Blind”
this paper focuses on color transformation methods for website images to
assist color blind individuals particularly red-green color blindness.

The study compares two algorithms for color transforma-tion: RGB to HSV
color space conversion and self-organizing color transformation, based
on criteria like ease of use, qual-ity, accuracy, and efficiency. The
research aims to enhance website images to meet the vision requirements
of color blind individuals by transforming colors into visible color
spaces.

The focus of the study is primarily on enhancing color visibility for
color blind individuals in website images, poten-tially overlooking
other aspects of color blindness challenges in different contexts. We
are including different types of color blindness favouring mojority of
the people.

> III\. EYE SIGHTEDNESS SIMULATION

Using python and OpenCV library to simulate visual defects such as
myopia (nearsightedness) and hypermetropia (farsight-edness) for an
image. It receives as input an input folder containing images, an output
folder to save the simulated images, and the user-specified parameters
for spherical power, cylindrical power, axis orientation, and the number
of images to be processed. The script builds on the primary
function-ality of the simulation of eyesight using the simulate_vd
function. This function receives an input image and applies both
spherical and cylindrical corrections according to the provided
parameters. The image is blurred with a Gaussian kernel, where the
kernel size depends on the absolute value of the spherical power, to
simulate myopia. The blurring effect simulates the fact that nearsighted
individuals cannot focus on distant objects. The image is scaled using a
focal length calculated from the spherical power to simulate
hypermetropia. The scale simulates the problem of focusing on nearby
objects in farsighted individuals. We then rescale the image back to its
original size using linear interpolation. A Gabor kernel is produced
according to the provided cylindrical power and axis orientation for
cylindrical correction. The Gabor kernel is a linear filter used for
edge detection and texture analysis. We convolve an image with the Gabor
kernel to simulate astigmatism-induced directional blurring.

> Fig. 2. Simulated Image

The source image depicted in Figure 1 is obtained from the Flickr image
dataset. Figure 2 describes a synthetic image of vision impairment: a
negative sphere of -3.0 diopters, a cylinder of -2.0 diopters, and an
axis aligned at 178 degrees.

> IV\. EYE SIGHT CORRECTION

<img src="./czosfz3p.png"
style="width:1.99973in;height:1.00827in" />Visually correcting eye
defects such as myopia and hy-permetropia within images using the OpenCV
library and NumPy for image processing was explored and using
pa-rameters like spherical power, cylindrical power, and axis
orientation to correct the visual defects. The core function for
correcting the visually defective eyesight is the function
apply_correction. First, we use the dimensions of the image and create a
mesh grid of the coordinates using the NumPy library. Then, the
spherical and cylindrical power values are converted from diopters to
meters by multiplying their values by -0.001. This must be done because
the correc-tions are to be done in the right units. Then, the
cylindrical power and axis orientation are also converted to radians to
perform mathematical operations. Then, the function computes the
distorted coordinates by subtracting the centre of the image from the
mesh grid coordinates. Then, these distorted coordinates are used to
calculate the corrected coordinates based on the spherical power,
cylindrical power, and axis orientation. The corrected coordinates are
then adjusted for the distortion caused by the visually defective
eyesight. After adjusting the corrected coordinates, they are mapped
using the cv2. remap function from OpenCV to get the corrected image.

> <img src="./emir4pfs.png" style="width:2in;height:1.328in" />Fig.
> 3. Original Image
>
> This mapping function maps the pixels of the original image to the
> corrected coordinates, hence applying the visual correction. Then the
> Canny edge detection algorithm is applied using the cv2. Canny
> algorithm to enhance the edges in the corrected image. Canny. The
> resulting edges are then
>
> Fig. 1. Original Image multiplied with a weight factor to control the
> strength of

<img src="./zvjipom2.png" style="width:2in;height:1.328in" />

the edge enhancement. Then the weighted edges are con-verted to a
3-channel image using cv2.bitwise_and to retain the original colours of
the corrected image. Weighted edges are finally combined with the
corrected image using cv2.addWeighted to create the final corrected
image with enhanced edges. The blending is done by adding the corrected
image with the weighted edges, along with a specified weight factor.

Sometimes, red and yellow colours are confused. Tritanopia, a severe
type of blue-yellow color blindness, simulates it by setting the blue
and green channels to the average of the red channel. This leads to the
confusion of blue and green colors. Monochromacy, or complete colour
blindness, is imitated by expanding the RGB channels of the image array
into three channels to be compatible with colour formats and then
summing the weighted average of the RGB channels to convert the image to
a grayscale image.

By modifying the RGB channels of the image array, each type of colour
blindness is simulated by altering the distribu-tion of colour
information in the image to be closer to the perceptual effects of
people with that particular kind of colour vision deficiency.

> <img src="./ohdbluop.png"
> style="width:1.99973in;height:1.00827in" />Fig. 4. Original Image
>
> <img src="./ofagan3z.png" style="width:2in;height:1.328in" />Fig.
> 6. colour blindness interface
>
> <img src="./eleiyyjs.png" style="width:2in;height:1.328in" />Fig.
> 5. Vision Corrected Image

Figure 3 is taken from the Flickr image dataset. Figure 4 shows the
correction of sphere -2.0, cylinder -1.0, and axis 178. We have
promising results, but we still need to implement super image resolution
to get clear images. Note that digital correction cannot replace
physical correction, such as lenses,c completely but it can aid up to a
level where viewing is comdortable.

> V. SIMULATING COLOUR BLINDNESS

<img src="./z5y11a3v.png" style="width:2in;height:1.328in" />Fig.
7. Original Image

PIL and NumPy for image processing is used to simulate several types of
color blindness. Each type of color blindness is processed by image
processing tailored to the specific type: Deuteranomaly, in which green
light sensitivity is lower, is simulated by adding a fraction of the
green channel with the red channel, shifting green colours toward the
red colour. Protanomaly, in which the sensitivity to a red light is
reduced, is performed by adjusting the red channel such that the overall
brightness is reduced to simulate the loss of ability to identify red
colour by people who suffer from protanomaly. Protanopia and
deuteranopia in which red and green colors are not distinguishable is
performed by averaging the red and green channels, thus removing the
ability to distinguish between red and green. Tritanomaly, which reduces
the sensitivity to blue and yellow lights, lowers the blue component of
the image such that blue and green colours become indistinguishable.

> Fig. 8. Color Blindness Simulation - Protanopia

Figure 5 is taken from the Flickr image dataset. Figure 6 shows the
simulation of protanopia colour blindness.

> VI\. AIDING COLOR BLINDNESS

Using OpenCV and scikit-learn for image processing and color detection,
a color detection model is made. The model aims to detect the top 5
dominant colours and then overlay the names of those colours on top of
the colour regions.

<img src="./axkuh0vc.png" style="width:2in;height:1.328in" /><img src="./oxud2vg4.png" style="width:2in;height:1.328in" />

The get_dominant_colors function is the core func-tion of the colour
detection part. This function takes a single image and the number of
dominant colours to detect, k, as its arguments. It then flattens the
image into a list of pixels. The list holds a 3-dimensional vector of
RGB values with each pixel. It uses the K-means clustering algorithms
from scikit-learn to extract the k dominant colours from the pixel
vector set. The coordinates of those clusters hold the RGB values of the
dominant colours.

The model also uses colour segmentation in the HSV colour space. This
colour space is more convenient for colour-based segmentation, so the
function flattens the image into the HSV colour space and then fixes the
lower and upper bounds for the targeted colour in the HSV colour space.
From these bounding boxes, a binary mask is generated, isolating the
regions that hold the targeted colour. In a thresholded image, the
image’s contours are detected, and the largest contour represents the
most influential region for the targeted color.

A dataset with RGB values for each colour is used for colour detection.
The model uses this .csv file to map RGB values of the dominant colour
to the colour’s name, and the mapping is done. Thus, all the detected
dominant colours can be associated with their closest matching colour
names with

the help of the dataset.

> VII\. WEBSITE AND BACKEND

The main aim of designing the website was to have a visual appeal and
user-friendly interface. HTML and CSS were se-lected as the main
technologies used in structuring the content and styling, giving the
website a clean and minimalistic look. The staticity of the pages was
chosen to achieve the goals of simplicity and efficiency because the
website’s main purpose was displaying information and allowing a user to
interact with the models.

Flask, a web framework, was chosen to achieve the goal of connecting web
pages and models. Flask tools would have allowed the development of a
strong API that will update the website in real time from the models.
From the Flask API, it is possible to address the models, send user
input, and receive generated output.

<img src="./kjua5g2h.png"
style="width:2.99985in;height:1.94831in" />When a user interacts with
the website by inputting custom values, the Flask API will perform the
request and transfer the relevant data to the models for processing.
After the models generate the output, for example, an image, the Flask
API will get the result and update the website dynamically in real-time.
That architecture ensures smooth workflow and efficiency, which allows
users to interact with the models through intuitive and visually
appealing interfaces.

> Fig. 9. Original Image
>
> Fig. 11. Opening Page
>
> <img src="./oxpktued.png"
> style="width:2.99985in;height:1.94831in" />Fig. 10. Displayed with
> colors

Figure 7 is taken from the Flickr image dataset. Figure 8 shows the
image with the most prominent colors being printed on top. This is not
the final result; the final result will have all the colours marked on
the spaces at their exact position. We are also working on the image
captioning model, which would be able to say what is present in the
image and can help in aiding people with colour blindness.

> Fig. 12. Options Page

All the above figures describe our website, which we designed to display
the results. The final website will be ready

<img src="./uerbcjgs.png"
style="width:2.99985in;height:1.94831in" />

> Fig. 13. Metrics Entry Page

by Milestone 3.

> VIII\. CONCLUSION

In this project we have created a software based solution that will
helped visually impaired users while using the website. We are using
Computer vision techniques like image wrapping, contranst enhancement,
edge detection, depth of field adjustments for real time rendering of
on-scree visuals. Adjusting the images accoding to the requirement of
people will have great impact on the users. There are few solutions
previously given to this problem but there are lot of practical
disadvantages in those solutions. There are few studies which focused on
this part but they are working only on the single problem this makes it
impractical for businesses to implement that in reality. We focuses on
different types of problems in a single method that will make us more
efficient. Although we’re more focused on making things more
visible—which is not directly associated with businesses at all—it will
surely contribute to increasing customer satisfaction. This is a process
that is very less effort-consuming compared to other solu-tions and is
standardizable into a template, applicable to any website. We have
presented best possible results developed, showcasing progress made in
the simulation and correction of visual defects, simulation of color
blindness, and aiding color-blind users through color detection and
labeling. We can adapt this to different websites very easily by
changing different images and can also include other problems in the
same technology This project will represent a giant leap toward
generating an inclusive and accessible digital environment that enables
equal access to digital content by all users, regardless of their visual
abilities. Solving this critical problem will empower blind people to
perceive the digital world without the need for external aids. We have
been promising so far, and we see promising upgrades in the coming, and
that is supposed to turn into a measure that is likely to advance the
web in making it more accessible and user-friendly to disabled persons.

> REFERENCES

1\. Fernandez-Bustos, J.-G., Gonzalez-Martı, I., Contreras, O., Cuevas,
R. (2022). Color Vision Deficiency Devices: A Sys-

tematic Review and Analysis. Health Science Reports, 5(9), e842.

2\. Lin, H.-Y., Chen, L.-Q., Wang, M.-L. (2019). Improving
Discrimination in Color Vision Deficiency by Image Re-Coloring. Sensors,
19(10), 2250.

3\. Fernandez-Bustos, J.-G., Gonzalez-Martı, I., Contreras, O., Cuevas,
R. (2017). A New Blur Measure for Depth Estimation. arXiv:1709.00072
\[cs.CV\].

4\. Gharbi, M., Chen, J., Barron, J. T., Hasinoff, S. W., Durand, F.
(2017). Deep Bilateral Learning for Real-Time Image Enhancement.
arXiv:1707.02880 \[cs.CV\].

5\. Kazemipour, A., Hadjistavrou, S., Favaro, P. (2017). Depth Defocus
and Blur Scale Estimation from Dense RGBD. Image and Vision Computing,
63, 47-57.

6\. Chakrabarti, A., Zickler, T., Freeman, W. T. (2010). Analyzing
spatially varying blur. In Proc. IEEE CVPR (pp. 2512–2519).

7\. Brettel, H., Vienot, F., Mollon, J. D. (1997). Comput-erized
simulation of color appearance for dichromats. Journal of the Optical
Society of America A, 14(12), 2647–2655.

8\. Lai, C. L., Chang, S. W. (2008). An image Processing Based Visual
Compensation System for Vision Defects. In International Symposium on
IEEE Xplore (pp. 472-476).

9\. Ching, S. L., Sabudin, M. (2010). A study of color transformation on
website images for the color blind. World Academy of Science,
Engineering and Technology, 38, 808-809.

10\. CHISNELL, D. REDISH, J.: Designing web sites for older adults:
Expert review of usability for older

11\. CHISNELL, D. REDISH, J.: Designing web sites for older adults: A
review of recent research. 2004.

12\. TULLIS, T. (2007): Older adults and the Web: Lessons learned from
Eye-Tracking. In: C. Stephanidis (ed.), Universal access in human
computer interaction. Coping with diversity. Reihe: Lecture Notes in
Computer Science (LNCS), 1030-1039, New York: Springer, 2007.

13.HARRISON, C.:Visual social semiotics: Understanding how still images
make meaning, in: Technical Communication, vol 50, no 1, 46-60 February
2003.

Prepared for AARP. 2004. Available at:
¡http://www.aarp.org/olderwiserwired¿. adults at 50 web sites..
Commissioned and delivered to Amy Lee, AARP. 2005. Available at:
¡http://www.aarp.org/olderwiserwired¿

