#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from matplotlib.ticker import AutoMinorLocator
from dotmap import DotMap

SERIES_NUMBER = 0
RENDERER = None

default_options = DotMap()

default_options.rcParams['figure.figsize'] = [8.0, 4.0]
default_options.rcParams['pdf.fonttype'] = 42
default_options.rcParams['ps.fonttype'] = 42
default_options.rcParams['font.size'] = 24
default_options.rcParams['font.family'] = 'Myriad Pro'
default_options.rcParams['text.color'] = 'gray'

default_options.x.axes.show = False
default_options.x.margin = 0.05
default_options.x.label.color = 'gray'
default_options.x.ticks.major.show = True
default_options.x.ticks.major.options.colors = 'gray'
default_options.x.ticks.minor.options.colors = 'gray'
default_options.x.grid.options.linestyle = 'dotted'
default_options.x.grid.options.linewidth = 0.5
default_options.x.grid.options.which = 'both'

default_options.y = default_options.x
default_options.y2 = default_options.x

default_options.legend.text.options.color = 'black'

default_options.broken.gap_position = 'bottom'

default_options.bar.width = 0.8

default_options.bar_labels.show = True
default_options.bar_labels.options.ha = 'center'
default_options.bar_labels.format_string = '%d'

default_options.vertical_lines.options.linestyle = '--'
default_options.horizontal_lines.options.linestyle = '--'

default_options.text.options.ha = 'center'
default_options.text.options.va = 'center'
default_options.text.options.color = 'black'


def merge_DotMap(a, b):
    for k, v in b.items():
        if isinstance(v, DotMap) and k in a:
            merge_DotMap(a[k], v)
        else:
            a[k] = v


def get_nth_dotmap(d, n):
    p = DotMap()
    for k, v in d.items():
        if isinstance(v, DotMap):
            p[k] = get_nth_dotmap(v, n)
        else:
            p[k] = v[n]
    return p


def autolabel(ax, rects, bar_labels):
    # attach some text labels
    # TODO: Make text smaller if it doesn't fit horizontally
    top = ax.get_ylim()[1]
    for i, rect in enumerate(rects):
        height = rect.get_height()
        _, y = rect.get_xy()
        a = ax.text(rect.get_x() + rect.get_width() / 2.0,
                    height - (0.005 * top) + y,
                    bar_labels.format_string % height,
                    va='top', color='white', **bar_labels.options.toDict())
        text_bbox = a.get_window_extent(renderer=RENDERER)
        rect_bbox = rect.get_window_extent(renderer=RENDERER)
        if text_bbox.y0 < rect_bbox.y0 or text_bbox.y1 > rect_bbox.y1:
            a.remove()
            ax.text(rect.get_x() + rect.get_width() / 2.0,
                    height + (0.005 * top) + y,
                    bar_labels.format_string % height,
                    va='bottom', color='black',
                    **bar_labels.options.toDict())


def plot_data(ax, x, y, options, series_options):
    global SERIES_NUMBER
    if options.plot_type == 'LINE':
        for i in xrange(len(x)):
            if len(x[i]) > 1:
                ax.plot(x[i], y[i], label=str(SERIES_NUMBER),
                        **series_options[i].toDict())
            else:
                ax.scatter(x[i], y[i], label=str(SERIES_NUMBER),
                           **series_options[i].toDict())
            SERIES_NUMBER += 1

    if options.plot_type == 'SCATTER':
        for i in xrange(len(x)):
            ax.scatter(x[i], y[i], label=str(SERIES_NUMBER),
                       **series_options[i].toDict())
            SERIES_NUMBER += 1

    elif options.plot_type == 'BAR' or options.plot_type == 'STACKED_BAR':
        rects = []
        for i in xrange(len(x)):
            b = np.sum(y[:i], 0) if options.plot_type == 'STACKED_BAR' else 0
            offset = SERIES_NUMBER * options.bar.width + \
                options.bar.width / 2.0 if options.plot_type == 'BAR' \
                else options.bar.width / 2.0
            rects += ax.bar(x[i] + offset, y[i], options.bar.width, bottom=b,
                            label=str(SERIES_NUMBER),
                            **series_options[i].toDict())
            SERIES_NUMBER += 1
        if options.bar_labels.show:
            autolabel(ax, rects, options.bar_labels)
        l = options.x.ticks.major.labels
        if len(l.locations) == 0:
            l.locations = np.arange(len(l.text)) + (0.8 / 2.0)

    if options.best_fit.show:
        for i in xrange(len(x)):
            m, b = np.polyfit(x[i], y[i], 1)
            ax.plot(x[i], [m*p + b for p in x[i]], linestyle='dotted',
                    label=str(SERIES_NUMBER), color='C%d' % i,
                    **options.best_fit.options[i].toDict())
            SERIES_NUMBER += 1


def apply_options_to_axis(axis, data, options):
    t = 'x' if (axis.axes.xaxis == axis) else 'y'

    axis.axes.set_axisbelow(True)
    axis.axes.margins(**{t: options.margin})

    if not options.axis.show:
        if t == 'x':
            axis.axes.spines['top'].set_visible(False)
            axis.axes.spines['bottom'].set_visible(False)
        else:
            axis.axes.spines['left'].set_visible(False)
            axis.axes.spines['right'].set_visible(False)
    
    if options.grid.show:
        axis.grid(**options.grid.options.toDict())

    if options.label['%slabel' % t]:
        getattr(axis.axes, 'set_%slabel' % t)(**options.label.toDict())

    if options.position:
        p = options.position
        axis.set_label_position(p)
        if p == 'right':
            axis.tick_right()
        if p == 'top':
            axis.tick_top()

    if options.log:
        if min(map(min, data)) < 0:
            getattr(axis.axes, 'set_%sscale' % t)('symlog')
        else:
            getattr(axis.axes, 'set_%sscale' % t)('log')

    if options.invert:
        getattr(axis.axes, 'invert_%saxis' % t)()

    if not options.ticks.major.show:
        options.ticks.major.options.length = 0
        getattr(axis.axes, 'set_%sticks' % t)([])

    if options.ticks.major.labels:
        l = options.ticks.major.labels
        if 'text' in l:
            if len(l.locations) == 0:
                l.locations = range(len(l.text))
            getattr(plt, '%sticks' % t)(l.locations,
                                        l.text, **l.options.toDict())
        elif 'locations' in l:
            getattr(plt, '%sticks' % t)(l.locations, **l.options.toDict())
        else:
            getattr(plt, '%sticks' % t)(**l.options.toDict())

    if options.ticks.major.options:
        axis.axes.tick_params(axis=t, which='major',
                              **options.ticks.major.options.toDict())

    if options.ticks.minor.show:
        if not options.log:
            axis.set_minor_locator(AutoMinorLocator(options.ticks.minor.count))
    else:
        options.ticks.minor.options.length = 0

    if options.ticks.minor.options:
        axis.axes.tick_params(axis=t, which='minor',
                              **options.ticks.minor.options.toDict())

    if options.limits:
        getattr(axis.axes, 'set_%slim' % t)(options.limits)


def plot_ax(ax, x, y, y2, options):
    if options.plot_type == 'BAR':
        if y2:
            options.bar.width /= len(y) + len(y2)
        else:
            options.bar.width /= len(y)
            
    plot_data(ax, x, y, options, options.series_options)
    apply_options_to_axis(ax.xaxis, x, options.x)
    apply_options_to_axis(ax.yaxis, y, options.y)

    if y2:
        ax2 = ax.twinx()
        plot_data(ax2, x, y2, options, options.series2_options)
        apply_options_to_axis(ax2.xaxis, x, options.x)
        apply_options_to_axis(ax2.yaxis, y2, options.y2)


def plot_broken(x, y, y2, options):
    global RENDERER, SERIES_NUMBER

    padding = 0.2
    if options.broken.yskip:
        bottom = [y[0][0] - padding, options.broken.yskip[0]]
        top = [options.broken.yskip[1] + 1, y[0][-1] + 1 + padding]
        options.broken.subplot.gridspec_kw = {'height_ratios':
                                              [top[1] - top[0],
                                               bottom[1] - bottom[0]]}
        new_options = deepcopy(options)
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True,
                                    **options.broken.subplot.toDict())
        new_options.y.limits = top
    else:
        left = [x[0][0] - padding, options.broken.xskip[0]]
        right = [options.broken.xskip[1] + 1, x[0][-1] + 1 + padding]
        options.broken.subplot.gridspec_kw = {'width_ratios':
                                              [left[1] - left[0],
                                               right[1] - right[0]]}
        new_options = deepcopy(options)
        f, (ax, ax2) = plt.subplots(1, 2, sharey=True,
                                    **options.broken.subplot.toDict())
        new_options.x.limits = left
    axes = [ax, ax2]
    RENDERER = f.canvas.get_renderer()

    # plot first axes
    new_options.legend.options.labels = []
    merge_DotMap(new_options, get_nth_dotmap(options.broken.options, 0))
    plt.sca(ax)
    plot_ax(ax, x, y, y2, new_options)

    SERIES_NUMBER = 0

    # plot second axes
    if options.broken.yskip:
        options.y.limits = bottom
        options.x.label.ylabel = ''
        options.x.ticks.major.show = False
        options.x.ticks.minor.show = False
    else:
        options.x.limits = right
        options.y.label.ylabel = ''
        options.y.ticks.major.show = False
        options.y.ticks.minor.show = False
    merge_DotMap(options, get_nth_dotmap(options.broken.options, 1))
    plt.sca(ax2)
    plot_ax(ax2, x, y, y2, options)
 
    # draw in 'gap' markers
    d = .015
    trans = ax.transScale + ax.transLimits
    d2 = {'middle': [0.5, 0.5], 'bottom': [0, 0], 'top': [1, 1],
          'zero': trans.transform([0, 0])}
    kwargs = dict(transform=ax.transAxes, color='gray', clip_on=False)
    if options.broken.yskip:
        d2 = d2[options.broken.gap_position][0]
        k1, k2 = options.broken.subplot.gridspec_kw['height_ratios']
        k = k1 / float(k2)
        ax.plot((d2-d, d2+d), (-d/k, +d/k), **kwargs)
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((d2-d, d2+d), (1-d, 1+d), **kwargs)
    else:
        d2 = d2[options.broken.gap_position][1]
        k1, k2 = options.broken.subplot.gridspec_kw['width_ratios']
        k = k1 / float(k2)
        ax.plot((1-d/k, 1+d/k), (d2-d, d2+d), **kwargs)
        kwargs.update(transform=ax2.transAxes)  # switch to the right axes
        ax2.plot((-d, +d), (d2-d, d2+d), **kwargs)

    return axes


def plot(x, y, my_options={}, y2=None):
    global RENDERER
    
    options = default_options.copy()
    merge_DotMap(options, my_options)
    
    plt.rcParams.update(options.rcParams.toDict())

    if options.broken.yskip or options.broken.xskip:
        axes = plot_broken(x, y, y2, options)
    else:
        f, ax = plt.subplots()
        RENDERER = f.canvas.get_renderer()
        axes = [ax]
        plot_ax(ax, x, y, y2, options)

    if options.legend.options.labels:
        handles, labels = axes[0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles),
                                      key=lambda t: t[0]))
        l = axes[-1].legend(handles=handles, **options.legend.options.toDict())
        for t in l.get_texts():
            t.update(options.legend.text.options.toDict())

    if options.vertical_lines.lines:
        for l in options.vertical_lines.lines:
            axes[0].axvline(l, **options.vertical_lines.options.toDict())

    if options.horizontal_lines.lines:
        for l in options.horizontal_lines.lines:
            axes[0].axhline(l, **options.horizontal_lines.options.toDict())
            
    for i in xrange(len(options.text.labels)):
        axes[1].text(*options.text.positions[i], s=options.text.labels[i],
                     transform=axes[0].transAxes,
                     **options.text.options.toDict())

    plt.tight_layout(pad=0)
    print options['output_fn']
    plt.savefig(options['output_fn'], bbox_inches='tight', pad_inches=0)
