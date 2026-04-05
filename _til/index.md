---
layout: default
title: Today I Learned
permalink: /til/
---

# Today I Learned

<ul>
  {% assign items = site.til | sort: "date" | reverse %}
  {% for item in items %}
    <li>
      <a href="{{ item.url | relative_url }}">{{ item.title }}</a>
    </li>
  {% endfor %}
</ul>
