import numpy as np
def hist_match(source, template, polynomial_order):
    source_copy = np.copy(source)
    template_copy = np.copy(template)

    train_hist = np.zeros((3, 256), dtype='int')
    target_hist = np.zeros((3, 256), dtype='int')

    for channel_ix in range(3):
        source = source_copy[:, :, channel_ix]
        template = template_copy[:, :, channel_ix]
        oldshape = source.shape

        # crop template to source shape
        l= oldshape[:2]
        # print(channel_ix, l)

        train_hist[channel_ix, :] += count_pixel_values(source)
        target_hist[channel_ix, :] += count_pixel_values(template)

    # convert histograms back to values for polynomial fitting
    coefficients = {}
    for channel_ix, channel in enumerate(['polyR', 'polyG', 'polyB']):
        train_values = []
        target_values = []
        for value_ix in range(256):
            train_values += [value_ix] * train_hist[channel_ix, value_ix]
            target_values += [value_ix] * target_hist[channel_ix, value_ix]
        assert len(train_values) == len(target_values)

        coefficients[channel] = np.polyfit(train_values, target_values, polynomial_order)

    matched = color_transform_polynoms(source_copy, coefficients['polyR'], coefficients['polyG'], coefficients['polyB'])
    return matched


def count_pixel_values(ary):
    """
    Helper for counting pixel values in a 256-color image
    """
    counts, _ = np.histogram(ary, bins=range(257))
    return counts


def color_transform_polynoms(img, polyR, polyG, polyB):
    """
    Performs a color transform on the rgb channels of a numpy array
    :param img: a numpy array, rgb color space
    :param polyR: polynom defining mapping on Red Channel, array of values see numpy.polynomial
    :param polyG: polynom defining mapping on Green Channel, array of values see numpy.polynomial
    :param polyB: polynom defining mapping on Blue Channel, array of values see numpy.polynomial
    :return: a numpy image array, RGB
    """

    mapRGB = np.zeros((3, 256))
    mapRGB[0] = np.polyval(polyR, np.arange(0, 256))
    mapRGB[1] = np.polyval(polyG, np.arange(0, 256))
    mapRGB[2] = np.polyval(polyB, np.arange(0, 256))
    mapRGB[mapRGB < 0] = 0
    mapRGB[mapRGB > 255] = 255

    new_img = img.copy()
    new_img[..., 0] = mapRGB[0][new_img[..., 0]]
    new_img[..., 1] = mapRGB[1][new_img[..., 1]]
    new_img[..., 2] = mapRGB[2][new_img[..., 2]]
    return np.uint8(new_img)


def ecdf(array_ravel):
    value, counts = np.unique(array_ravel, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64)
    cdf /= cdf[-1]
    return value, cdf


def histogram_matching(item_src, item_ref):
    x1, y1 = ecdf(item_src.ravel())
    x2, y2 = ecdf(item_ref.ravel())
    matched = hist_match(item_src, item_ref, 10)
    x3, y3 = ecdf(matched.ravel())
    matched_src = Image.fromarray(matched)
    return matched_src