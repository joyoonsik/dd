<template>
  <div id="my_dataviz"></div>
</template>

<script>
import * as d3 from "d3";

export default {
  name: 'LineChart',  
  props:['LineChartData'],
  created(){
      this.data = this.LineChartData
  },
  mounted(){
    // console.log(this.data);
    let margin = {
            top: 100,
            right: 900,
            bottom: 30,
            left: 170
        },
        width = 2560 - margin.left - margin.right,
        height = 250 - margin.top - margin.bottom;

    let svg = d3.select("#my_dataviz")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    let sumstat = d3.nest()
        .key(function(d) {
            return d.key;
        })
        .entries(this.data);

    let x = d3.scaleLinear()
        .domain([d3.min(this.data, function(d) {
            return +d.ymdhms;
        }), d3.max(this.data, function(d) {
            return +d.ymdhms;
        })])
        .range([0, width]);

    let xscale = d3.scaleTime()
        .domain([new Date(2020, 0, 1), new Date(2020, 11, 31)])
        .range([0, width]);

    let x_axis = d3.axisBottom(xscale)
        .tickFormat(d3.timeFormat("%b"));

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(x_axis);

    let y = d3.scaleLinear()
        .domain([d3.min(this.data, function(d) {
            return +d.pow;
        }), d3.max(this.data, function(d) {
            return +d.pow;
        })])
        .range([height, 0]);
    svg.append("g")
        .call(d3.axisLeft(y));

    let res = sumstat.map(function(d) {
        return d.key
    })
    let color = d3.scaleOrdinal()
        .domain(res)
        .range(['black'])

    svg.selectAll(".line")
        .data(sumstat)
        .enter()
        .append("path")
        .style("opacity", .2)
        .attr("fill", "none")
        .attr("stroke", function(d) {
            return color(d.key)
        })
        .attr("stroke-width", 1.5)
        .attr("d", function(d) {
            return d3.line()
                .x(function(d) {
                    return x(d.ymdhms);
                })
                .y(function(d) {
                    return y(+d.pow);
                }) (d.values)
        })

    for(let i=1; i<12; i++){
      svg.append('line')
        .attr('x1', width / 12*i)
        .attr('y1', 0)
        .attr('x2', width / 12*i)
        .attr('y2', height)
        .attr('stroke', 'red')
        .attr("stroke-width", 2)
    }
    
    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("text-decoration", "underline")
        .text("Graph Title");
  },
};
</script>