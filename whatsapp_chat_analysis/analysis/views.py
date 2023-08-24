from typing import Any, Dict

import pandas as pd
from django.forms.models import BaseModelForm
from django.http import HttpResponse
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView

from analysis.graph import get_sns_plot
from analysis.whatsap import whatsapp_open

from .forms import UserSearchForm
from .models import UserSearch

# Create your views here.


class Index(TemplateView):
    template_name = "base.html"


class Whatsapp_Chat(CreateView):
    template_name = "data.html"
    form_class = UserSearchForm
    queryset = UserSearch.objects.all()
    success_url = "success/"

    def form_valid(self, form: BaseModelForm) -> HttpResponse:
        username = form.cleaned_data["username"]
        range_scroll = form.cleaned_data["range_scroll"]
        print(type(username), range_scroll)
        whatsapp_open(username, range_scroll)

        return super().form_valid(form)


class Success_View(TemplateView):
    template_name = "success.html"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        df_friend = pd.read_csv(
            "/media/ahsan/2EB4AF30B4AEF98B/Final_Done_Complete/whatsapp_chat_analysis/Friend_chatting_new.csv"
        )
        df_feature = pd.read_csv(
            "/media/ahsan/2EB4AF30B4AEF98B/Final_Done_Complete/whatsapp_chat_analysis/Feature_Engineering.csv"
        )
        df_count = pd.read_csv(
            "/media/ahsan/2EB4AF30B4AEF98B/Final_Done_Complete/whatsapp_chat_analysis/after_count_vectorizer.csv"
        )
        df_tdidf = pd.read_csv(
            "/media/ahsan/2EB4AF30B4AEF98B/Final_Done_Complete/whatsapp_chat_analysis/after_tfidf_vectorizer.csv"
        )

        grapgh_1 = get_sns_plot(df_friend)
        # get_sns_plot(df_feature)
        # get_sns_plot(df_count)
        # get_sns_plot(df_tdidf)

        context["df_friend"] = df_friend.head().to_html()
        context["df_friend_shape"] = df_friend.describe().to_html()
        context["graph_1"] = grapgh_1
        context["df_feature"] = df_feature.head().to_html()
        context["df_feature_shape"] = df_feature.describe().to_html()
        # context["graph_2"] = grapgh_2
        context["df_count"] = df_count.head().to_html()
        context["df_count_shape"] = df_count.describe().to_html()
        # context["graph_3"] = grapgh_3
        context["df_tdidf"] = df_tdidf.head().to_html()
        context["df_tdidf_shape"] = df_count.describe().to_html()
        # context["graph_4"] = grapgh_4
        return context


class Whatsapp_Use_View(TemplateView):
    template_name = "whatsapp_use.html"
