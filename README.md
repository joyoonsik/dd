# pv-project

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Lints and fixes files
```
npm run lint
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).

# ------------------------추가------------------------

# How to draw d3.js multiple line chart in vue.js

**d3.js는 버전에 따라서 그래프를 그립니다.**

d3의 여러 버전을 관리하고 싶다면 
[여기를](https://stackoverflow.com/questions/16156445/multiple-versions-of-a-script-on-the-same-page-d3-js) 참고해주세요.

LineChart 컴포넌트 에서는 버전4만 사용하기 때문에 npm i d3뒤에 @4를 추가했습니다.
### Install

```bash
npm i d3@4
```

### Usage

Import:

```javascript
import * as d3 from "d3";
```
### LineChart.vue Code Explain

Template:

```html
<div id="my_dataviz"></div>
```
javascript:
```javascript
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
  }
```

### 그래프를 그리는 방식
1. `created`에서 `props`를 통해 데이터(json format)를 전달 받습니다.
2. `mounted`기능을 통해 그래프가 그려집니다.

### 추가한 Vue Component 구조

### 추가한 컴포넌트간 데이터 송수신

### Result
- 두번째 세번째 그래프는 데이터가 하드코딩 되어있습니다. 

![image](https://user-images.githubusercontent.com/50390923/107531604-5bbf5e00-6c00-11eb-8d07-0b5dbb9ee883.png)

## 참고한 사이트 

- [Data Visualization with Vue and D3](https://alligator.io/vuejs/visualization-vue-d3/)
- [d3-graph-gallery](https://www.d3-graph-gallery.com/index.html)
- [vue.js 기능들](https://joshua1988.github.io/web-development/vuejs/vuejs-tutorial-for-beginner/)


