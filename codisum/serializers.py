from rest_framework import serializers


class DiffSerializer(serializers.Serializer):
    diff = serializers.CharField(max_length=1000000)

    def update(self, instance, validated_data):
        pass

    def create(self, validated_data):
        pass


class MsgSerializer(serializers.Serializer):
    msg = serializers.CharField(max_length=1000)
    score = serializers.CharField(max_length=20)

    def update(self, instance, validated_data):
        pass

    def create(self, validated_data):
        pass
