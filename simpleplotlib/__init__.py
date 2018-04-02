#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from copy import deepcopy
from cycler import cycler
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from dotmap import DotMap

SERIES_NUMBER = 0
RENDERER = None

default_options = DotMap()

default_options.rcParams['figure.figsize'] = [8.0, 4.0]
default_options.rcParams['pdf.fonttype'] = 42
default_options.rcParams['ps.fonttype'] = 42
default_options.rcParams['font.size'] = 24
default_options.rcParams['font.family'] = 'Myriad Pro'
default_options.rcParams['text.color'] = 'black'

default_options.x.axis.show = False
default_options.x.axis.color = 'gray'
default_options.x.margin = 0.05
default_options.x.label.color = 'black'
default_options.x.ticks.major.show = True
default_options.x.ticks.major.options.colors = 'black'
default_options.x.ticks.minor.options.colors = 'black'
default_options.x.ticks.minor.count = 5
default_options.x.grid.options.linestyle = 'dotted'
default_options.x.grid.options.linewidth = 0.5
default_options.x.grid.options.which = 'both'

default_options.y = default_options.x
default_options.y2 = default_options.x

default_options.legend.text.options.color = 'black'

default_options.broken.gap_positions = ['bottom']

default_options.bar.width = 0.8

default_options.bar_labels.show = True
default_options.bar_labels.options.ha = 'center'
default_options.bar_labels.format_string = '%d'

default_options.vertical_lines.options.linestyle = '--'
default_options.horizontal_lines.options.linestyle = '--'
default_options.annotation_lines.options.linestyle = '--'

default_options.text.options.ha = 'center'
default_options.text.options.va = 'center'
default_options.text.options.color = 'black'

default_options.inset.options.location = 'lower right'
default_options.inset.options.zoom_level = 1.5
default_options.inset.options.border_color = 'black'
default_options.inset.options.corners = [1, 3]
default_options.inset.marker.options.color = 'gray'

default_options.vertical_shaded.options.alpha = 0.5
default_options.vertical_shaded.options.color = 'red'
default_options.horizontal_shaded = default_options.vertical_shaded


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
    if 'color_groups' in options.series.toDict():
        cm = plt.get_cmap('tab20c')
        max_color = 4
        current = [0, 0, 0, 0, 0]
        for i in xrange(len(x)):
            group = options.series.color_groups[i]
            series_options[i].color = cm(group * max_color + current[group])
            current[group] = (current[group] + 1) % max_color
    if 'yerr' in options.series_options[0].toDict():
        options.plot_type = 'ERROR'
    if options.plot_type == 'LINE':
        for i in xrange(len(x)):
            if 'color' not in series_options[i].toDict():
                series_options[i].color = 'C%d' % i
            if len(x[i]) > 1:
                ax.plot(x[i], y[i], label=str(SERIES_NUMBER),
                        **series_options[i].toDict())
            else:
                if 'markersize' in options.series_options[i].toDict():
                    options.series_options[i].s = \
                        options.series_options[i].markersize
                    del options.series_options[i].markersize
                ax.scatter(x[i], y[i], label=str(SERIES_NUMBER),
                           **series_options[i].toDict())
            SERIES_NUMBER += 1

    if options.plot_type == 'ERROR':
        for i in xrange(len(x)):
            if len(x[i]) > 1:
                ax.errorbar(x[i], y[i], label=str(SERIES_NUMBER),
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

    if t == 'x':
        axis.axes.spines['top'].set_visible(False)
        axis.axes.spines['bottom'].set_visible(False)
    else:
        axis.axes.spines['left'].set_visible(False)
        axis.axes.spines['right'].set_visible(False)

    if options.axis.show:
        if t == 'x':
            if options.position and options.position == 'top':
                sp = axis.axes.spines['top']
            else:
                sp = axis.axes.spines['bottom']
        else:
            if options.position and options.position == 'right':
                sp = axis.axes.spines['right']
            else:
                sp = axis.axes.spines['left']
        sp.set_visible(True)
        sp.set_color(options.axis.color)

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

    if options.y.label_offset:
        (yx, yy) = ax.yaxis.get_label().get_position()
        ax.yaxis.set_label_coords(yx + options.y.label_offset[0],
                                  yy + options.y.label_offset[1])

    if options.x.label_offset:
        (xx, xy) = ax.xaxis.get_label().get_position()
        ax.xaxis.set_label_coords(xx + options.x.label_offset[0],
                                  xy + options.x.label_offset[1])

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
    d = .03  # .015
    trans = ax.transScale + ax.transLimits
    d_dict = {'middle': [0.5, 0.5], 'bottom': [0, 0], 'top': [1, 1],
              'zero': trans.transform([0, 0])}
    kwargs = dict(transform=ax.transAxes, color='black', clip_on=False)
    if options.broken.yskip:
        k1, k2 = options.broken.subplot.gridspec_kw['height_ratios']
        k = k1 / float(k2)
        for p in options.broken.gap_positions:
            d2 = d_dict[p][0]
            kwargs.update(transform=ax.transAxes)
            ax.plot((d2-d, d2+d), (-d/k, +d/k), **kwargs)
            # switch to the bottom axes
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((d2-d, d2+d), (1-d, 1+d), **kwargs)
    else:
        k1, k2 = options.broken.subplot.gridspec_kw['width_ratios']
        k = k1 / float(k2)
        for p in options.broken.gap_positions:
            d2 = d_dict[p][1]
            kwargs.update(transform=ax.transAxes)
            ax.plot((1-d/k, 1+d/k), (d2-d, d2+d), **kwargs)
            # switch to the right axes
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d, +d), (d2-d, d2+d), **kwargs)
    return axes


def plot(x, y, my_options={}, y2=None):
    global RENDERER, SERIES_NUMBER
    SERIES_NUMBER = 0
    
    options = default_options.copy()
    merge_DotMap(options, my_options)
    
    plt.rcParams.update(options.rcParams.toDict())

    if options.broken.yskip or options.broken.xskip:
        axes = plot_broken(x, y, y2, options)
    else:
        f, ax = plt.subplots()
        RENDERER = f.canvas.get_renderer()
        axes = [ax]
        if len(x) > 10:
            cm = plt.get_cmap('tab20')
            ax.set_prop_cycle(cycler('color',
                                     [cm(1.*i/20) for i in range(20)]))

        if 'styles' in options.legend.toDict() and \
           'labels' in options.legend.options.toDict():
            dummy_lines = []
            for i, style in enumerate(options.legend.styles):
                if 'color' not in style:
                    style.color = 'C%d' % i
                dummy_lines.append(mlines.Line2D([], [], **style.toDict()))
            l = ax.legend(handles=dummy_lines,
                          **options.legend.options.toDict())
            for t in l.get_texts():
                t.update(options.legend.text.options.toDict())
        plot_ax(ax, x, y, y2, options)

    plt.tight_layout(pad=0)

    if options.inset.show:
        locations = {
            'upper right': 1,
            'upper left': 2,
            'lower left': 3,
            'lower right': 4,
            'right': 5,
            'center left': 6,
            'center right': 7,
            'lower center': 8,
            'upper center': 9,
            'center': 10,
        }
        ax_inset = zoomed_inset_axes(axes[0], options.inset.options.zoom_level,
                                     loc=locations[
                                         options.inset.options.location])
        SERIES_NUMBER = 0
        del options.x.label
        del options.y.label
        options.x.limits = options.inset.options.x.limits
        options.y.limits = options.inset.options.y.limits
        plot_ax(ax_inset, x, y, y2, options)
        for sp in [ax_inset.axes.spines[i]
                   for i in ['top', 'bottom', 'left', 'right']]:
            sp.set_visible(True)
            sp.set_color(options.inset.options.border_color)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        plt.setp(ax_inset, xticks=[], yticks=[])
        mark_inset(axes[0], ax_inset, loc1=options.inset.options.corners[0],
                   loc2=options.inset.options.corners[1], fc="none",
                   ec=options.inset.options.marker.options.color)

        plt.draw()

    if options.x.axis.stretch:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         box.width * options.x.axis.stretch, box.height])

    if options.y.axis.stretch:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         box.width, box.height * options.y.axis.stretch])

    if 'styles' not in options.legend.toDict() and \
       'labels' in options.legend.options.toDict():
        handles, labels = axes[0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles),
                                      key=lambda t: int(t[0])))
        if options.legend.order:
            handles = [h
                       for _, h in sorted(zip(options.legend.order, handles))]
            options.legend.options.labels = \
                [h for _, h in sorted(zip(options.legend.order,
                                          options.legend.options.labels))]
        l = axes[-1].legend(handles=handles,
                            **options.legend.options.toDict())
        for t in l.get_texts():
            t.update(options.legend.text.options.toDict())

    if options.vertical_lines.lines:
        for l in options.vertical_lines.lines:
            axes[0].axvline(l, **options.vertical_lines.options.toDict())

    if options.vertical_shaded.limits:
        for vs in options.vertical_shaded.limits:
            l, r = vs[:2]
            o = options.vertical_shaded.options.toDict()
            if len(vs) > 2:
                o = vs[2].toDict()
            axes[0].axvspan(l, r, **o)

    if options.horizontal_lines.lines:
        for l in options.horizontal_lines.lines:
            axes[0].axhline(l, **options.horizontal_lines.options.toDict())

    if options.horizontal_shaded.limits:
        for hs in options.horizontal_shaded.limits:
            l, r = hs[:2]
            o = options.horizontal_shaded.options.toDict()
            if len(hs) > 2:
                o = hs[2].toDict()
            axes[0].axhspan(l, r, **o)

    for i in xrange(len(options.text.labels)):
        axes[-1].text(*options.text.positions[i], s=options.text.labels[i],
                      transform=axes[0].transAxes,
                      **options.text.options.toDict())

    for i in options.annotation_lines.lines:
        axes[0].annotate('', xy=i[0], xytext=i[1], arrowprops=dict(
            arrowstyle='-', **options.annotation_lines.options.toDict()))

    print options['output_fn']
    plt.savefig(options['output_fn'], bbox_inches='tight', pad_inches=0)
