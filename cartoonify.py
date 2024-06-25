#################################################################
# FILE : cartoonify.py
# WRITER : shimon ifrach , simaon 78, 211553219
# EXERCISE : intro2cse ex6 2022
# DESCRIPTION: doing a cartoonified picture
# PEOPLE YOU DISCUSSED:pnina perry
#################################################################
from ex6_helper import *
import math
import sys


def separate_channels(image):
    """this function are getting a matrix with rgb type colors and separate
                        it to channels with only one color in every channel"""
    third_d = []
    for k in range(len(image[0][0])):
        second_d = []
        # referring the inner values in the matrix by using
        #                   three loops to reach the pixels
        for i in range(len(image)):
            first_d = []
            for j in range(len(image[0])):
                first_d.append(image[i][j][k])
            # reorganize
            second_d.append(first_d)
        third_d.append(second_d)
    return third_d


def combine_channels(channels):
    """
    this function are getting a separated matrix into
                                    channels and combine it to rgb module
    """
    third_d = []
    for j in range(len(channels[0])):
        # using three loops to reach the value inside the 3d matrix
        second_d = []
        for k in range(len(channels[0][0])):
            first_d = []
            for i in range(len(channels)):
                first_d.append(channels[i][j][k])
            second_d.append(first_d)
        third_d.append(second_d)
    return third_d


def RGB2grayscale(colored_image):
    """this function are converting full 3d image and returns a gray image"""
    second_d = []

    for i in range(len(colored_image)):
        first_d = []
        for j in range(len(colored_image[i])):
            red_color = colored_image[i][j][0]
            green_color = colored_image[i][j][1]
            blue_color = colored_image[i][j][2]
            d_red = red_color * 0.299
            d_green = green_color * 0.587
            d_blue = blue_color * 0.114
            sum_all = round(d_red + d_green + d_blue)
            # using a specific formila
            first_d.append(sum_all)
        second_d.append(first_d)
    return second_d


def blur_kernel(size):
    """this function are getting an odd natural number and converting it
    into a kernel by creating a square with frictions of one over num square"""
    optimized_size = 1 / (size ** 2)
    second_d = []
    for j in range(size):
        # adding the frictions into a new kernel sized matrix
        first_d = []
        for i in range(size):
            first_d.append(optimized_size)
        second_d.append(first_d)
    return second_d


def apply_kernel(image, kernel):
    """this function are apply the kernel by multiplying it into the pixels
                        neighborhood and dividing it with an average formula"""
    region = 0
    second_d = []
    for i in range(len(image)):
        first_d = []
        for j in range(len(image[i])):
            for x in range(len(kernel)):
                for y in range(len(kernel[x])):
                    # after getting into the specific value in the matrix we
                    # using a formula to apply the kernel over the number
                    if i - len(kernel) // 2 + x < 0 or i - len(
                            kernel) // 2 + x > \
                            len(image) - 1 or j - len(kernel) // 2 + y > \
                            len(image[i]) - 1 or j - len(kernel) // 2 + y < 0:
                        region = image[i][j] * kernel[x][y] + region
                    else:
                        region = image[i - len(kernel) // 2 + x][
                                     j - len(kernel) // 2 + y] \
                                 * kernel[x][y] + region
            # returning it as 0 or 255
            if region > 255:
                region = 255
            elif region < 0:
                region = 0
            first_d.append(round(region))
            region = 0
        second_d.append(first_d)
    return second_d


def for_y_in_the_edge(image, x, y):
    """this function are using a bilinear interpolation
    function by the case of y value in the edge of the pixel"""
    c_x_value = int(x + 1 // 1)
    a_x_value = int(x // 1)
    c_final = image[int(y)][c_x_value]
    a_final = image[int(y)][a_x_value]
    # delta x formula
    return round(a_final * (1 - (x - (x // 1))) + c_final * (x - (x // 1)))


def for_x_in_the_edge(image, x, y):
    """this function are calling the billinear interpolation
    function for the case of x value in the pixels edge"""
    c_y_value = int(y + 1 // 1)
    d_y_value = int(y // 1)
    c_final = image[c_y_value][int(x)]
    d_final = image[d_y_value][int(x)]  # calling for indexes
    # delta y formula
    return round(c_final * (y - (y // 1)) + d_final * (1 - (y - (y // 1))))


def for_none_in_the_edges(image, x, y):
    """this function are calling the billinear interpolation function for
     the case of none x and y on the edge of a pixel"""
    a_x_value = int(x // 1)
    a_y_value = int(y // 1)
    b_x_value = int(x // 1)
    b_y_value = int(y + 1 // 1)
    c_x_value = int(x + 1 // 1)
    c_y_value = int(y // 1)
    d_x_value = int(x + 1 // 1)
    d_y_value = int(y + 1 // 1)
    a_final = image[a_y_value][a_x_value]
    b_final = image[b_y_value][b_x_value]
    c_final = image[c_y_value][c_x_value]
    d_final = image[d_y_value][d_x_value]
    # delta calculation with x and y
    return round(
        a_final * (1 - (x - (x // 1))) * (1 - (y - (y // 1))) + b_final *
        (y - (y // 1)) * (1 - (x - (x // 1))) + c_final
        * (x - (x // 1)) * (1 - (y - (y // 1))) + d_final * (x - (x // 1)) *
        (y - (y // 1)))


def bilinear_interpolation(image, y, x):
    """this function are doing a bilinear
    interpolation over a specific resized picture"""
    if int(x // 1) != x and int(y // 1) != y:
        return for_none_in_the_edges(image, x, y)
    elif int(x // 1) == x and int(y // 1) != y:
        return for_x_in_the_edge(image, x, y)
    elif int(y // 1) == y and int(x // 1) != x:
        return for_y_in_the_edge(image, x, y)
    elif int(y // 1) == y and int(x // 1) == x:
        return image[int(y)][int(x)]
    # calling help functions


def resize(image, new_height, new_width):
    """
    this function are resizing picture to new sized picture
    and keeping the proportion
    """
    second_d = []
    for i in range(int(new_height)):
        first_d = []
        new_i = (i / (new_height - 1)) * (len(image) - 1)
        for j in range(int(new_width)):
            new_j = (j / (new_width - 1)) * (len(image[0]) - 1)
            # using the resize formula
            final_image = bilinear_interpolation(image, new_i, new_j)
            first_d.append(final_image)
        second_d.append(first_d)

    return second_d


def for_x_y_exeeded(image, max_size):
    """this function are using the colored image resize
    function with x and y in exceeded"""
    lst = []
    faze_1 = separate_channels(image)
    # doing separation before resizing it
    for i in range(len(faze_1)):
        faze_2 = resize(faze_1[i], max_size, len(image[0]) / (len(image)
                                                              / max_size))
        lst.append(faze_2)
    faze_3 = combine_channels(lst)
    return faze_3


def for_x_exeeded(image, max_size):
    """this function are using the colored image
    resize function with x exceeded """
    lst = []
    faze_1 = separate_channels(image)
    for i in range(len(faze_1)):
        faze_2 = resize(faze_1[i], len(image) / (len(image[0]) / max_size),
                        max_size)
        lst.append(faze_2)
    faze_3 = combine_channels(lst)
    return faze_3


def for_y_exeeded(image, max_size):
    """this function are using the colored image resize
    function with y only exceeded"""
    faze_1 = separate_channels(image)
    lst = []
    for i in range(len(faze_1)):
        faze_2 = resize(faze_1[i], max_size, len(image[0]) / (len(image)
                                                              / max_size))
        lst.append(faze_2)
    faze_3 = combine_channels(lst)
    return faze_3


def scale_down_colored_image(image, max_size):
    """this function are calling all of the helper functions
    and resizing a colored image"""
    if max_size >= len(image) and max_size >= len(image[0]):
        return None
    elif len(image) > max_size and not len(image[0]) > max_size:
        faze_3 = for_y_exeeded(image, max_size)
        return faze_3
    elif len(image[0]) > max_size and not len(image) > max_size:
        faze_3 = for_x_exeeded(image, max_size)
        return faze_3
    elif len(image[0]) > max_size and len(image) > max_size:
        faze_3 = for_x_y_exeeded(image, max_size)
        return faze_3


def rotate_90(image, direction):
    """this function are taking picture and rotating it in 90
     degrees to the right or to the left"""
    second_d = []
    if direction == 'L':
        for j in range(len(image[0]) - 1, -1, -1):
            # for the left turn
            first_d = []
            for i in range(len(image)):
                first_d.append(image[i][j])
            second_d.append(first_d)
        return second_d
    elif direction == 'R':
        # for the right turn
        for j in range(len(image[0])):
            first_d = []
            for i in range(len(image) - 1, -1, -1):
                first_d.append(image[i][j])
            second_d.append(first_d)
        return second_d


def get_edges(image, blur_size, block_size, c):
    """this function are getting the edges in a color in a picture
    and converting it into black color over white"""
    blur_function = blur_kernel(blur_size)
    blur_image = apply_kernel(image, blur_function)
    r = block_size // 2
    threshold = 0
    second_d = []
    for i in range(len(blur_image)):
        # doing a similar formula like kernel
        first_d = []
        for j in range(len(blur_image[i])):
            for x in range(block_size):
                for y in range(block_size):
                    if i - r + x < 0 or i - r + x > \
                            len(blur_image) - 1 or j - r + y > \
                            len(blur_image[i]) - 1 or j - r + y < 0:
                        threshold = (blur_image[i][j]) + threshold
                    else:
                        threshold = (blur_image[i - r + x][j - r + y]) \
                                    + threshold
            if (threshold / block_size ** 2) - c > image[i][j]:
                threshold = 0
            else:
                threshold = 255
            first_d.append(round(threshold))
            threshold = 0
        second_d.append(first_d)
    return second_d


def quantize(image, N):
    """this function are doing a quantization on a picture witch is to delete
     the difference between pixels"""
    second_d = []
    for i in range(len(image)):
        first_d = []
        for j in range(len(image[i])):
            # getting into the inside value in the picture
            quantized_pix = round(
                math.floor(image[i][j] * N / 256) * 255 / (N - 1))
            first_d.append(quantized_pix)
        second_d.append(first_d)
    return second_d


def quantize_colored_image(image, N):
    """this function are doing the same quantization
     but over a colored image"""
    separated_image = separate_channels(image)
    third_d = []
    for i in range(len(separated_image)):
        # getting into the inner value
        quantized_colored_separated_image = quantize(separated_image[i], N)
        third_d.append(quantized_colored_separated_image)
    ready_quantized_image = combine_channels(third_d)
    return ready_quantized_image


def for_3d_list(image1, image2, mask):
    """this function are adding the picture with quantized picture
    and a mask witch is a edged picture """
    third_d = []
    separated_3d_image1 = separate_channels(image1)
    separated_3d_image2 = separate_channels(image2)
    for i in range(len(separated_3d_image1)):
        # getting into the inner value in the 3d matrix
        second_d = []
        for j in range(len(separated_3d_image2[i])):
            first_d = []
            for k in range(len(separated_3d_image2[i][j])):
                masked_image = round(
                    separated_3d_image1[i][j][k] * mask[j][k] +
                    separated_3d_image2[i][j][k] * (1 - mask[j][k]))
                first_d.append(masked_image)
            second_d.append(first_d)
        third_d.append(second_d)
    masked_final = combine_channels(third_d)
    return masked_final


def for_2d_list(image1, image2, mask):
    """this function are adding mask but for three d list"""
    second_d = []
    for i in range(len(image1)):
        first_d = []
        for j in range(len(image1[i])):
            masked_image = round(
                image1[i][j] * mask[i][j] +
                image2[i][j] * (1 - mask[i][j]))
            first_d.append(masked_image)
        second_d.append(first_d)
    return second_d


def add_mask(image1, image2, mask):
    """this function are adding the mask witch is is the edged picture by
     calling the other three previous functions"""
    if type(image2[0][0]) != int:
        masked_final = for_3d_list(image1, image2, mask)
        return masked_final
    else:
        second_d = for_2d_list(image1, image2, mask)
        return second_d


def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    """this function are responsible for running the entire program with
    calling all of the previous functions"""
    quantized_pic = quantize_colored_image(image, quant_num_shades)
    black_and_white = RGB2grayscale(image)
    edged_picture = get_edges(black_and_white, blur_size, th_block_size, th_c)
    separated_picture = separate_channels(quantized_pic)
    second_d = []
    for i in range(len(edged_picture)):
        # getting inside the inner value in the 3d matrix
        first_d = []
        for j in range(len(edged_picture[i])):
            mask = edged_picture[i][j] / 255
            first_d.append(mask)
        second_d.append(first_d)
    third_d = []
    for k in range(len(separated_picture)):
        add_masked_separated_pic = add_mask(separated_picture[k],
                                            edged_picture, second_d)
        third_d.append(add_masked_separated_pic)
    catoonify_final = combine_channels(third_d)
    return catoonify_final


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print('INVALID PARAMETERS THE CORRECT NUMBER IS 8!!')
    else:
        loaded_image = load_image(sys.argv[1])
        max_size = int(sys.argv[3])
        scled_down_image = scale_down_colored_image(loaded_image, max_size)
        if scled_down_image is None:
            scled_down_image = loaded_image
        cartoonified_image = cartoonify(scled_down_image, int(sys.argv[4]),
                                        int(sys.argv[5]), int(sys.argv[6]),
                                        int(sys.argv[7]))
        save_image(cartoonified_image, sys.argv[2])
